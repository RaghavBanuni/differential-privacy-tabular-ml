"""Mechanisms and accounting: the arithmetic that makes "private" a measurable claim.

**The definition.** A randomised mechanism ``M`` is ``(eps, delta)``-differentially private if for all datasets
``D``, ``D'`` differing in one record and all measurable ``S``:

    P(M(D) in S) <= exp(eps) * P(M(D') in S) + delta

Read it as a bound on what any adversary can learn about one person's participation -- whatever their side
information, and whatever they do with the output afterwards. It is a property of the *mechanism*, not of the
data, which is why it survives post-processing and composition.

**Sensitivity is the whole game.** Every mechanism here needs the amount a single record can change the query:

    Delta_1 f = max over neighbouring D, D' of || f(D) - f(D') ||_1     (Laplace)
    Delta_2 f = max over neighbouring D, D' of || f(D) - f(D') ||_2     (Gaussian)

For a count, ``Delta_1 = 1``. For a **sum over an unbounded domain, sensitivity is unbounded** -- an average
salary has no finite sensitivity unless salaries are clipped first, and the clipping bound is then part of the
privacy analysis rather than a preprocessing detail. Getting sensitivity wrong is the most common way a "DP"
system provides no privacy at all: the noise is added, the epsilon is reported, and the guarantee is void.

**Composition.** Privacy loss accumulates. Basic composition adds epsilons, which is correct and pessimistic.
Advanced composition gives roughly ``sqrt(k)`` growth, and Renyi DP composes exactly by *addition of RDP
epsilons*, then converts to ``(eps, delta)`` once at the end -- which is why RDP is what DP-SGD accounting uses.
The three are implemented side by side because the difference between them decides whether a training run is
reportable or absurd.

**Amplification by subsampling.** A Gaussian mechanism applied to a random ``q``-fraction of the data is much
more private than the same mechanism on all of it: a record not sampled leaks nothing. The RDP bound for the
subsampled Gaussian is what makes DP-SGD possible at all, and the bound implemented here is Mironov's
closed-form expression for the Poisson-subsampled Gaussian, valid at integer orders.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------------------------
# mechanisms
# ---------------------------------------------------------------------------------------------


def laplace_noise(scale: float, rng: random.Random) -> float:
    """Laplace(0, scale) by inverse transform: ``-scale * sign(u) * log(1 - 2|u|)``, u uniform on (-0.5, 0.5)."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    uniform = rng.random() - 0.5
    return -scale * math.copysign(1.0, uniform) * math.log(1.0 - 2.0 * abs(uniform))


def laplace_mechanism(
    value: float, sensitivity: float, epsilon: float, rng: random.Random
) -> float:
    """``f(D) + Lap(Delta_1 / eps)``: pure ``(eps, 0)``-DP.

    No delta, so no failure probability -- the guarantee holds absolutely rather than with high probability.
    That is worth something in a regulated setting, and it costs more noise per unit of epsilon than Gaussian
    does for multi-dimensional queries.
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive; a query with zero sensitivity needs no noise")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return value + laplace_noise(sensitivity / epsilon, rng)


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical Gaussian calibration: ``sigma >= Delta_2 * sqrt(2 ln(1.25/delta)) / eps``.

    Valid only for ``eps < 1``; above that the classical bound is not merely loose, it is **wrong**, so this
    raises rather than returning a number that looks fine. The analytic Gaussian mechanism (Balle & Wang, 2018)
    removes the restriction and is the right implementation for production use.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError(
            "the classical Gaussian bound requires eps < 1; use the analytic Gaussian mechanism or "
            "RDP accounting instead of extrapolating it"
        )
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_mechanism(
    value: float, sensitivity: float, epsilon: float, delta: float, rng: random.Random
) -> float:
    """``f(D) + N(0, sigma^2)`` with ``sigma`` from the classical calibration."""
    return value + rng.gauss(0.0, gaussian_sigma(sensitivity, epsilon, delta))


def randomised_response(truth: bool, epsilon: float, rng: random.Random) -> bool:
    """Local DP for a yes/no question: answer truthfully with probability ``e^eps / (1 + e^eps)``.

    The local model needs no trusted curator and pays for it: estimating a proportion to within a percentage
    point at ``eps = 1`` takes on the order of a hundred thousand respondents. Apple and Google deploy local DP
    because they cannot promise to be trustworthy; a hospital analysing its own records can use the central
    model and get far more utility for the same epsilon.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    truthful_probability = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    return truth if rng.random() < truthful_probability else not truth


def debias_randomised_response(positive_share: float, epsilon: float) -> float:
    """Invert randomised response to recover the underlying proportion.

    The estimate is unbiased and can fall outside [0, 1] on small samples. That is not a bug: clamping would
    introduce bias, and a proportion of -0.03 is honest evidence that the sample is too small for the chosen
    epsilon.
    """
    truthful = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    return (positive_share - (1.0 - truthful)) / (2.0 * truthful - 1.0)


def clipped_mean(
    values: "list[float]", lower: float, upper: float, epsilon: float, rng: random.Random
) -> float:
    """A private mean, done properly: clip to a public range, then add noise scaled to that range.

    The sensitivity of a mean over ``n`` records with values in ``[lower, upper]`` is ``(upper - lower) / n`` --
    finite **only because of the clipping**. Two consequences usually skipped:

    * the clipping range must come from public knowledge or be paid for with privacy budget; choosing it by
      looking at the data's min and max leaks the extremes, which are exactly the most identifiable records;
    * clipping introduces bias, and on a heavy-tailed variable that bias can exceed the noise. A private mean
      income computed with a clipping bound of 100k is not an estimate of mean income.
    """
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    if not values:
        raise ValueError("no values")
    clipped = [min(max(value, lower), upper) for value in values]
    sensitivity = (upper - lower) / len(values)
    return laplace_mechanism(sum(clipped) / len(clipped), sensitivity, epsilon, rng)


def private_histogram(
    labels: "list[str]", categories: "tuple[str, ...]", epsilon: float, rng: random.Random
) -> "dict[str, float]":
    """A histogram over a **public** category list: sensitivity 1, one Laplace draw per bucket.

    The category list must be public. Deriving the buckets from the data means the *presence of a bucket* leaks
    a record -- the classic failure where a histogram of diagnoses reveals that someone in the dataset has a
    rare disease, no matter how much noise the count carries.

    Counts may go negative after noise. Reporting them as-is keeps the estimate unbiased; clamping to zero is
    valid post-processing but biases every small bucket upwards, which matters precisely where the data is
    sparse.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    counts = dict.fromkeys(categories, 0)
    for label in labels:
        if label in counts:
            counts[label] += 1
    return {
        category: count + laplace_noise(1.0 / epsilon, rng) for category, count in counts.items()
    }


# ---------------------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------------------


def basic_composition(epsilons: "list[float]", deltas: "list[float]") -> "tuple[float, float]":
    """Epsilons add, deltas add. Always valid, and pessimistic for many compositions."""
    return sum(epsilons), sum(deltas)


def advanced_composition(
    epsilon: float, delta: float, k: int, target_delta: float
) -> "tuple[float, float]":
    """``k``-fold composition (Dwork-Rothblum-Vadhan):

        eps_total = sqrt(2 k ln(1/delta')) * eps + k * eps * (e^eps - 1)

    Roughly ``sqrt(k)`` rather than ``k`` for small epsilon, which is the difference between a thousand-step
    training run being reportable and being nonsense. It *adds* a ``delta'``: privacy that holds with high
    probability rather than absolutely.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1)")
    total = math.sqrt(2.0 * k * math.log(1.0 / target_delta)) * epsilon + k * epsilon * (
        math.exp(epsilon) - 1.0
    )
    return total, k * delta + target_delta


# ---------------------------------------------------------------------------------------------
# Renyi DP
# ---------------------------------------------------------------------------------------------

DEFAULT_ORDERS = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64)


def gaussian_rdp(alpha: float, noise_multiplier: float) -> float:
    """RDP of the Gaussian mechanism: ``alpha / (2 sigma^2)``.

    Linear in ``alpha``, inversely quadratic in ``sigma``. The elegance is that composition becomes addition:
    ``k`` steps cost ``k * alpha / (2 sigma^2)`` at each order, with no approximation at all.
    """
    if alpha <= 1:
        raise ValueError("RDP order must exceed 1")
    if noise_multiplier <= 0:
        raise ValueError("noise multiplier must be positive")
    return alpha / (2.0 * noise_multiplier**2)


def subsampled_gaussian_rdp(alpha: int, noise_multiplier: float, sample_rate: float) -> float:
    """RDP of the Poisson-subsampled Gaussian at integer order ``alpha`` (Mironov et al., 2019):

        eps(alpha) = (1 / (alpha - 1)) * log( sum_{j=0}^{alpha} C(alpha, j) (1-q)^{alpha-j} q^j
                                              * exp( j (j-1) / (2 sigma^2) ) )

    This bound is why DP-SGD works: sampling a 1% minibatch makes each step roughly a hundred times cheaper in
    privacy terms, because a record that was not sampled cannot have influenced the update.

    Restricted to integer orders, where the closed form is valid. A fractional order needs numerical
    integration, and silently rounding to an integer would misstate the guarantee. At ``q = 1`` the sum
    collapses to the single ``j = alpha`` term and reproduces the plain Gaussian bound exactly, which the tests
    check -- it is the natural consistency condition on this formula.
    """
    if alpha != int(alpha) or alpha < 2:
        raise ValueError("this closed form requires an integer order alpha >= 2")
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must lie in (0, 1]")
    if noise_multiplier <= 0:
        raise ValueError("noise multiplier must be positive")

    alpha = int(alpha)
    if sample_rate == 1.0:
        return gaussian_rdp(alpha, noise_multiplier)

    log_terms = []
    for j in range(alpha + 1):
        log_binomial = math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
        log_terms.append(
            log_binomial
            + (alpha - j) * math.log1p(-sample_rate)
            + (j * math.log(sample_rate) if j else 0.0)
            + (j * (j - 1)) / (2.0 * noise_multiplier**2)
        )

    # log-sum-exp: the exponent grows quadratically in j, so naive summation overflows for large alpha
    largest = max(log_terms)
    total = largest + math.log(sum(math.exp(term - largest) for term in log_terms))
    return total / (alpha - 1)


def rdp_to_dp(rdp_epsilon: float, alpha: float, delta: float) -> float:
    """``eps = rdp + log(1/delta) / (alpha - 1)``.

    The standard conversion. Tighter variants exist (Canonne-Kamath-Steinke, Balle et al.) and give a visibly
    smaller epsilon for the same run. Using the loose one is safe -- it over-states the loss -- but a headline
    epsilon depends on the accountant as much as on the noise, which is worth saying out loud when comparing
    numbers between papers.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return rdp_epsilon + math.log(1.0 / delta) / (alpha - 1.0)


@dataclass
class RDPAccountant:
    """Accumulate RDP across steps at several orders, convert once, and take the best order.

    Optimising over orders at the end is not a trick: RDP composes exactly at every order, so the tightest
    available ``(eps, delta)`` statement is the minimum over them. Fixing one order in advance throws away a
    factor that can be substantial.
    """

    orders: "tuple[int, ...]" = DEFAULT_ORDERS
    spent: "dict[int, float]" = field(default_factory=dict)
    steps: int = 0

    def __post_init__(self) -> None:
        self.spent = {order: 0.0 for order in self.orders}

    def step(self, noise_multiplier: float, sample_rate: float, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be positive")
        for order in self.orders:
            self.spent[order] += count * subsampled_gaussian_rdp(
                order, noise_multiplier, sample_rate
            )
        self.steps += count

    def epsilon(self, delta: float) -> "tuple[float, int]":
        candidates = [
            (rdp_to_dp(value, order, delta), order) for order, value in self.spent.items()
        ]
        return min(candidates, key=lambda item: item[0])

    def report(self, delta: float) -> str:
        epsilon, order = self.epsilon(delta)
        return f"{self.steps} steps -> (eps = {epsilon:.3f}, delta = {delta:g}) at RDP order {order}"


def noise_for_target_epsilon(
    target_epsilon: float,
    delta: float,
    steps: int,
    sample_rate: float,
    orders: "tuple[int, ...]" = DEFAULT_ORDERS,
    tolerance: float = 1e-4,
) -> float:
    """Bisect for the noise multiplier that hits a target epsilon -- the calibration people actually need.

    Epsilon decreases monotonically in the noise multiplier, so bisection is exact to tolerance. This is the
    right direction to run the calculation: fix the privacy budget as a policy decision, then discover what it
    costs in accuracy. Choosing the noise first and reporting whatever epsilon falls out is how "eps = 8" ends
    up in a paper without anyone having decided that eps = 8 was acceptable.
    """
    if target_epsilon <= 0:
        raise ValueError("target epsilon must be positive")

    def epsilon_for(sigma: float) -> float:
        accountant = RDPAccountant(orders)
        accountant.step(sigma, sample_rate, steps)
        return accountant.epsilon(delta)[0]

    low, high = 0.3, 2.0
    while epsilon_for(high) > target_epsilon:
        high *= 2.0
        if high > 4096.0:
            raise ValueError("cannot reach the target epsilon: reduce steps or the sample rate")
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if epsilon_for(middle) > target_epsilon:
            low = middle
        else:
            high = middle
    return high
