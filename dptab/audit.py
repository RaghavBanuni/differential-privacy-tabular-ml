"""Auditing: what the epsilon actually buys, measured with an attack rather than argued from theory.

Epsilon is an upper bound on privacy loss. It says nothing about how much a *particular* adversary can learn
from a *particular* model, and the gap between the two is usually enormous -- a model trained at eps = 8 may be
almost impossible to attack, while an "eps = infinity" model leaks individual records to a five-line script.
Reporting only the epsilon is therefore either alarmism or complacency, depending on which side of the gap you
happen to be standing.

Three measurements here:

**Membership inference** (Shokri et al., 2017; Yeom et al., 2018). A model that memorised a record assigns it a
lower loss than a comparable record it never saw. Thresholding the per-example loss is the simplest attack that
works, and its advantage -- ``TPR - FPR`` at the best threshold -- is a lower bound on real leakage. An attack
that succeeds is proof of a problem; an attack that fails is *not* proof of safety, only evidence that this
attack failed, and stronger attacks (shadow models, LiRA) do considerably better.

**Empirical epsilon.** Any attack achieving ``(TPR, FPR)`` implies a lower bound on the true epsilon
(Jagielski et al., 2020):

    eps_empirical >= log( (1 - delta - FPR) / FNR )     and symmetrically

If a measured lower bound ever exceeded the accountant's epsilon, the implementation would be provably broken.
That comparison is the most useful test in this repository: it does not verify the guarantee -- no finite
experiment can -- but it can *falsify* it.

**Disparate impact** (Bagdasaryan et al., 2019). DP-SGD costs the minority group more accuracy than the
majority. The mechanism is clipping: under-represented patterns produce large gradients for longer, so their
contributions are clipped hardest, and noise then buries what remains. "Privacy for everyone" is paid for
unequally, and the gap is measured here per group rather than reported as an aggregate that hides it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .learning import Dataset, LogisticModel


@dataclass
class AttackResult:
    """A membership-inference attack, reported as a bound rather than a verdict."""

    auc: float
    advantage: float
    best_threshold: float
    tpr: float
    fpr: float
    canary_advantage: float | None = None

    @property
    def empirical_epsilon(self) -> float:
        """A lower bound on epsilon implied by the attack's operating point.

        Uses ``log((1 - FPR) / FNR)`` and its mirror, taking the larger. Returns 0.0 when the attack is at or
        below chance, since an attack that cannot distinguish members implies no lower bound at all -- not a
        negative one.
        """
        fnr = 1.0 - self.tpr
        candidates = []
        if fnr > 1e-9 and self.fpr < 1.0:
            candidates.append(math.log((1.0 - self.fpr) / max(fnr, 1e-9)))
        if self.tpr > 1e-9 and self.fpr > 1e-9:
            candidates.append(math.log((1.0 - fnr) / max(self.fpr, 1e-9)))
        best = max(candidates) if candidates else 0.0
        return max(best, 0.0)

    def line(self) -> str:
        canary = "" if self.canary_advantage is None else f"   canary adv {self.canary_advantage:+.3f}"
        return (
            f"attack AUC {self.auc:.3f}   advantage {self.advantage:+.3f}   "
            f"TPR {self.tpr:.3f} @ FPR {self.fpr:.3f}   eps_lower {self.empirical_epsilon:.2f}{canary}"
        )


def _auc(member_scores: "list[float]", nonmember_scores: "list[float]") -> float:
    """Rank AUC of the attack: probability a member gets a *lower* loss than a non-member."""
    if not member_scores or not nonmember_scores:
        return float("nan")
    wins = 0.0
    for member in member_scores:
        for nonmember in nonmember_scores:
            wins += 1.0 if member < nonmember else (0.5 if member == nonmember else 0.0)
    return wins / (len(member_scores) * len(nonmember_scores))


def membership_inference(
    model: LogisticModel, train: Dataset, holdout: Dataset
) -> AttackResult:
    """Threshold the per-example loss: predict "member" when the loss is below a threshold.

    Both sets are drawn from the same distribution, so any separation is memorisation rather than a
    distributional artefact -- an attack evaluated against a holdout from a *different* distribution measures
    covariate shift and calls it privacy leakage, which is a common and flattering mistake.
    """
    member_losses = [model.loss(record) for record in train.records]
    nonmember_losses = [model.loss(record) for record in holdout.records]

    auc = _auc(member_losses, nonmember_losses)
    thresholds = sorted(set(member_losses + nonmember_losses))
    if len(thresholds) > 400:  # subsample the threshold grid on large datasets
        step = len(thresholds) // 400
        thresholds = thresholds[::step]

    best = (0.0, 0.0, 0.0, 0.0)  # advantage, threshold, tpr, fpr
    for threshold in thresholds:
        tpr = sum(1 for loss in member_losses if loss <= threshold) / len(member_losses)
        fpr = sum(1 for loss in nonmember_losses if loss <= threshold) / len(nonmember_losses)
        if tpr - fpr > best[0]:
            best = (tpr - fpr, threshold, tpr, fpr)

    canary_advantage = None
    canary_losses = [model.loss(record) for record in train.records if record.canary]
    if canary_losses:
        ordinary = [model.loss(record) for record in train.records if not record.canary]
        canary_advantage = (
            sum(ordinary) / len(ordinary) - sum(canary_losses) / len(canary_losses)
        )

    return AttackResult(auc, best[0], best[1], best[2], best[3], canary_advantage)


def canary_exposure(model: LogisticModel, train: Dataset, holdout: Dataset) -> float:
    """How much better the model fits planted canaries than genuinely unseen records.

    Canaries are unique and labelled against the pattern, so no generalisable rule explains them. A positive
    value means the model stored individual records; near zero means it did not, which is what DP is for.
    """
    canaries = [model.loss(record) for record in train.records if record.canary]
    if not canaries:
        raise ValueError("this dataset has no canaries")
    unseen = [model.loss(record) for record in holdout.records]
    return sum(unseen) / len(unseen) - sum(canaries) / len(canaries)


@dataclass
class GroupImpact:
    """Per-group accuracy under a private and a non-private model, and the gap between the two."""

    baseline: "dict[str, float]"
    private: "dict[str, float]"

    def cost(self) -> "dict[str, float]":
        return {
            group: self.baseline[group] - self.private.get(group, 0.0) for group in self.baseline
        }

    def disparity(self) -> float:
        """How much more accuracy the worst-hit group loses than the best-hit one.

        The number that an aggregate accuracy figure hides. A model that loses two points overall may have lost
        one point on the majority and fifteen on a minority group, and the aggregate will not say so.
        """
        costs = self.cost()
        return max(costs.values()) - min(costs.values()) if costs else 0.0

    def report(self) -> str:
        lines = [f"{'group':<12} {'baseline':>9} {'private':>9} {'cost':>8}"]
        for group, baseline in sorted(self.baseline.items()):
            private = self.private.get(group, float("nan"))
            lines.append(f"{group:<12} {baseline:>9.3f} {private:>9.3f} {baseline - private:>8.3f}")
        lines.append(f"\ndisparity between best- and worst-hit group: {self.disparity():.3f}")
        return "\n".join(lines)


def group_impact(
    baseline: LogisticModel, private: LogisticModel, evaluation: Dataset
) -> GroupImpact:
    return GroupImpact(baseline.group_accuracy(evaluation), private.group_accuracy(evaluation))


def audit_bound_is_consistent(attack: AttackResult, accounted_epsilon: float) -> bool:
    """The falsification test: an empirical lower bound above the accountant's epsilon means a broken run.

    A finite experiment cannot verify a DP guarantee. It can refute one, and this is how -- which makes it worth
    running on every implementation change, since the failure it catches (a mis-stated sensitivity, a sampling
    scheme that does not match the amplification bound) is otherwise completely silent.
    """
    return attack.empirical_epsilon <= accounted_epsilon + 1e-9


def utility_privacy_curve(
    dataset: Dataset,
    epsilons: "tuple[float, ...]" = (0.5, 1.0, 2.0, 4.0, 8.0),
    delta: float = 1e-5,
    seed: int = 0,
) -> "list[dict]":
    """Train at several budgets and report accuracy, attack success and the per-group cost at each.

    This is the table a privacy decision should be made from. Epsilon alone does not say what it costs, and
    accuracy alone does not say what it buys -- and neither says who pays.
    """
    from .learning import train_dpsgd, train_sgd

    train, holdout = dataset.split(0.7, seed=seed)
    baseline = train_sgd(train, seed=seed)
    baseline_attack = membership_inference(baseline.model, train, holdout)

    rows = [
        {
            "epsilon": None,
            "accuracy": baseline.model.accuracy(holdout),
            "attack_auc": baseline_attack.auc,
            "advantage": baseline_attack.advantage,
            "empirical_epsilon": baseline_attack.empirical_epsilon,
            "clip_rate": 0.0,
            "groups": baseline.model.group_accuracy(holdout),
        }
    ]
    for epsilon in epsilons:
        result = train_dpsgd(train, target_epsilon=epsilon, delta=delta, seed=seed)
        attack = membership_inference(result.model, train, holdout)
        rows.append(
            {
                "epsilon": result.epsilon,
                "accuracy": result.model.accuracy(holdout),
                "attack_auc": attack.auc,
                "advantage": attack.advantage,
                "empirical_epsilon": attack.empirical_epsilon,
                "clip_rate": result.clip_rate,
                "groups": result.model.group_accuracy(holdout),
            }
        )
    return rows
