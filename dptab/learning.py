"""Data, logistic regression, and DP-SGD -- with the per-example clipping done honestly.

DP-SGD (Abadi et al., 2016) is four changes to ordinary SGD:

1. **Per-example gradients.** Not a batch gradient. Every record's gradient is computed separately, because the
   next step bounds each record's individual influence.
2. **Clip each one to L2 norm C.** This is what makes sensitivity finite: with per-example gradients bounded by
   ``C`` and Poisson sampling, one record can move the summed gradient by at most ``C``.
3. **Sum, add Gaussian noise ``N(0, (sigma C)^2 I)``, then divide by the expected batch size.** Noise is added
   to the *sum*, so the noise per averaged coordinate falls as the batch grows -- which is why DP-SGD wants
   large batches, the opposite of the usual advice.
4. **Account for the privacy loss** of every step with RDP over the subsampled Gaussian.

Two implementation details that are usually where a DP-SGD implementation quietly stops being private:

* **Poisson sampling, not shuffling.** The amplification bound assumes each record is included independently
  with probability ``q``. Fixed-size shuffled minibatches -- what every framework does by default -- do not
  satisfy it, and the resulting epsilon is understated. This implementation samples by independent coin flips,
  so batch sizes vary, and divides by the *expected* batch size rather than the realised one (dividing by the
  realised count would itself be a data-dependent quantity and leak).
* **Clipping is not the same as gradient-norm regularisation.** Clipping *biases* the update towards the
  examples whose gradients are small, which is the mechanism behind DP's disparate impact on minority groups:
  under-represented patterns produce large gradients for longer, so their contributions are the ones clipped
  hardest and buried in noise. ``audit.py`` measures this rather than describing it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .mechanisms import RDPAccountant


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    features: "tuple[float, ...]"
    label: int
    group: str = "majority"
    canary: bool = False


@dataclass(frozen=True)
class Dataset:
    records: "tuple[Record, ...]"
    feature_names: "tuple[str, ...]"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dimension(self) -> int:
        return len(self.records[0].features)

    def split(self, fraction: float = 0.7, seed: int = 0) -> "tuple[Dataset, Dataset]":
        rng = random.Random(seed)
        shuffled = list(self.records)
        rng.shuffle(shuffled)
        cut = int(fraction * len(shuffled))
        return (
            Dataset(tuple(shuffled[:cut]), self.feature_names),
            Dataset(tuple(shuffled[cut:]), self.feature_names),
        )

    def group_counts(self) -> "dict[str, int]":
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.group] = counts.get(record.group, 0) + 1
        return counts


def make_dataset(
    n: int = 4000,
    minority_share: float = 0.08,
    canaries: int = 20,
    seed: int = 0,
) -> Dataset:
    """A tabular binary-classification problem with a minority subgroup and planted canaries.

    Three deliberate features:

    * The **minority group** (``minority_share`` of the data) follows a *different* decision rule. A model has
      to spend capacity on it, and DP-SGD's clipping plus noise is what stops it from doing so -- which is the
      disparate-impact effect, and it is measurable here because the group label is known.
    * **Canaries** are records with an extreme, unique feature signature and a surprising label. They are the
      standard instrument for memorisation: a model that fits them has memorised individuals rather than
      learned a pattern, and a membership-inference attack finds them first.
    * Features are on comparable scales, because per-example gradient clipping is **not scale invariant** --
      an unscaled feature dominates the gradient norm, absorbs the clipping budget, and quietly ruins the
      model. Scaling is part of the privacy pipeline, not preprocessing hygiene.
    """
    rng = random.Random(seed)
    names = ("age", "income", "tenure", "usage", "region")
    records: list[Record] = []
    minority_count = int(n * minority_share)

    for index in range(n):
        minority = index < minority_count
        age = rng.gauss(0.0, 1.0)
        income = rng.gauss(0.0, 1.0)
        tenure = rng.gauss(0.0, 1.0)
        usage = rng.gauss(0.0, 1.0)
        region = rng.gauss(0.0, 1.0)
        if minority:
            # a genuinely different rule: the signs on income and usage are reversed
            logit = 0.4 * age - 1.3 * income + 0.6 * tenure - 1.1 * usage + 0.2 * region
        else:
            logit = 0.9 * age + 1.2 * income - 0.5 * tenure + 0.8 * usage + 0.1 * region
        probability = 1.0 / (1.0 + math.exp(-logit))
        records.append(
            Record(
                features=(age, income, tenure, usage, region),
                label=1 if rng.random() < probability else 0,
                group="minority" if minority else "majority",
            )
        )

    for canary_index in range(canaries):
        # unique, extreme, and labelled against the pattern: only memorisation can fit these
        signature = tuple(
            4.0 if position == canary_index % 5 else -3.5 for position in range(5)
        )
        records.append(Record(features=signature, label=canary_index % 2, group="canary", canary=True))

    rng.shuffle(records)
    return Dataset(tuple(records), names)


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass
class LogisticModel:
    """Plain logistic regression with a bias, and the gradient written out.

    For one example, ``grad = (sigmoid(w.x + b) - y) * [x, 1]``. Keeping this explicit is the point: DP-SGD
    needs the *per-example* gradient, and any framework trick that computes only the batch mean has already
    destroyed the quantity the privacy analysis is about.
    """

    weights: "list[float]"
    bias: float = 0.0

    @classmethod
    def zeros(cls, dimension: int) -> "LogisticModel":
        return cls([0.0] * dimension, 0.0)

    def logit(self, features: "tuple[float, ...]") -> float:
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def predict(self, features: "tuple[float, ...]") -> float:
        return sigmoid(self.logit(features))

    def gradient(self, record: Record) -> "list[float]":
        error = self.predict(record.features) - record.label
        return [error * value for value in record.features] + [error]

    def loss(self, record: Record) -> float:
        """Per-example log loss, clamped away from log(0). Used by the model and by the attacker."""
        probability = min(max(self.predict(record.features), 1e-12), 1.0 - 1e-12)
        return -(
            record.label * math.log(probability) + (1 - record.label) * math.log(1.0 - probability)
        )

    def accuracy(self, dataset: Dataset) -> float:
        correct = sum(
            1 for record in dataset.records if (self.predict(record.features) >= 0.5) == bool(record.label)
        )
        return correct / len(dataset)

    def group_accuracy(self, dataset: Dataset) -> "dict[str, float]":
        totals: dict[str, list[int]] = {}
        for record in dataset.records:
            bucket = totals.setdefault(record.group, [0, 0])
            bucket[1] += 1
            if (self.predict(record.features) >= 0.5) == bool(record.label):
                bucket[0] += 1
        return {group: hits / count for group, (hits, count) in totals.items()}

    def auc(self, dataset: Dataset) -> float:
        """Rank-based AUC, computed exactly (ties counted as half)."""
        scores = [(self.predict(record.features), record.label) for record in dataset.records]
        positives = [score for score, label in scores if label == 1]
        negatives = [score for score, label in scores if label == 0]
        if not positives or not negatives:
            return float("nan")
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                wins += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
        return wins / (len(positives) * len(negatives))


def clip_l2(gradient: "list[float]", bound: float) -> "tuple[list[float], bool]":
    """Scale a gradient to L2 norm at most ``bound``; report whether it was clipped.

    The clip rate is the single most useful diagnostic in DP-SGD. Near 100% means the bound is far too small and
    the update direction is being determined by clipping rather than by data; near 0% means the bound is too
    large and the noise (which scales with ``C``) is larger than it needs to be.
    """
    if bound <= 0:
        raise ValueError("clipping bound must be positive")
    norm = math.sqrt(sum(value**2 for value in gradient))
    if norm <= bound:
        return gradient, False
    scale = bound / norm
    return [value * scale for value in gradient], True


@dataclass
class TrainingResult:
    model: LogisticModel
    epsilon: float | None
    delta: float | None
    steps: int
    clip_rate: float
    loss_curve: "list[float]" = field(default_factory=list)
    accountant: RDPAccountant | None = None

    @property
    def private(self) -> bool:
        return self.epsilon is not None

    def summary(self) -> str:
        privacy = (
            f"eps = {self.epsilon:.2f} at delta = {self.delta:g}" if self.private else "no privacy guarantee"
        )
        return f"{self.steps} steps, {privacy}, clipped {self.clip_rate:.1%} of gradients"


def train_sgd(
    dataset: Dataset,
    learning_rate: float = 0.3,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 0,
) -> TrainingResult:
    """Ordinary minibatch SGD: the utility ceiling, and an entirely non-private baseline.

    Included because "the DP model scores 0.78" means nothing without it. The gap to this number is the price of
    the guarantee, and reporting the price is the whole point of the exercise.
    """
    rng = random.Random(seed)
    model = LogisticModel.zeros(dataset.dimension)
    records = list(dataset.records)
    curve: list[float] = []
    steps = 0

    for _ in range(epochs):
        rng.shuffle(records)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            summed = [0.0] * (dataset.dimension + 1)
            for record in batch:
                for index, value in enumerate(model.gradient(record)):
                    summed[index] += value
            for index in range(dataset.dimension):
                model.weights[index] -= learning_rate * summed[index] / len(batch)
            model.bias -= learning_rate * summed[-1] / len(batch)
            steps += 1
        curve.append(sum(model.loss(record) for record in records) / len(records))

    return TrainingResult(model, None, None, steps, 0.0, curve)


def train_dpsgd(
    dataset: Dataset,
    target_epsilon: float | None = 3.0,
    delta: float = 1e-5,
    noise_multiplier: float | None = None,
    clip_bound: float = 1.0,
    learning_rate: float = 0.5,
    epochs: int = 20,
    expected_batch_size: int = 256,
    seed: int = 0,
) -> TrainingResult:
    """DP-SGD with per-example clipping, Poisson sampling, and RDP accounting.

    Either give ``target_epsilon`` (the noise multiplier is then calibrated to it, which is the right direction)
    or give ``noise_multiplier`` directly and read off the epsilon that results.

    The update, precisely:

        g_i     = clip(grad_i, C)               for each sampled i
        g_noisy = (sum_i g_i + N(0, (sigma C)^2 I)) / E[batch size]
        w      <- w - lr * g_noisy

    Note the division by the **expected** batch size, not the realised one. Poisson sampling makes the realised
    count a random function of the data, so dividing by it would leak -- a small detail that invalidates the
    guarantee, and exactly the kind of detail that separates a DP implementation from a DP-flavoured one.
    """
    if clip_bound <= 0:
        raise ValueError("clip_bound must be positive")
    rng = random.Random(seed)
    sample_rate = min(expected_batch_size / len(dataset), 1.0)
    steps_per_epoch = max(int(1.0 / sample_rate), 1)
    total_steps = epochs * steps_per_epoch

    if noise_multiplier is None:
        if target_epsilon is None:
            raise ValueError("give either target_epsilon or noise_multiplier")
        from .mechanisms import noise_for_target_epsilon

        noise_multiplier = noise_for_target_epsilon(
            target_epsilon, delta, total_steps, sample_rate
        )

    model = LogisticModel.zeros(dataset.dimension)
    accountant = RDPAccountant()
    dimension = dataset.dimension + 1
    clipped_count = 0
    seen = 0
    curve: list[float] = []

    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            # Poisson sampling: independent inclusion, so the batch size varies. This is what the
            # amplification bound assumes, and what shuffled fixed-size batches do not satisfy.
            batch = [record for record in dataset.records if rng.random() < sample_rate]
            summed = [0.0] * dimension
            for record in batch:
                gradient, was_clipped = clip_l2(model.gradient(record), clip_bound)
                clipped_count += int(was_clipped)
                seen += 1
                for index, value in enumerate(gradient):
                    summed[index] += value

            noisy = [
                (value + rng.gauss(0.0, noise_multiplier * clip_bound)) / (sample_rate * len(dataset))
                for value in summed
            ]
            for index in range(dataset.dimension):
                model.weights[index] -= learning_rate * noisy[index]
            model.bias -= learning_rate * noisy[-1]
            accountant.step(noise_multiplier, sample_rate)

        curve.append(sum(model.loss(record) for record in dataset.records) / len(dataset))

    epsilon, _ = accountant.epsilon(delta)
    return TrainingResult(
        model,
        epsilon,
        delta,
        accountant.steps,
        clipped_count / max(seen, 1),
        curve,
        accountant,
    )
