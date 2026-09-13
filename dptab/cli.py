"""Demos.

    python -m dptab.cli mechanisms    Laplace, Gaussian, randomised response, and sensitivity
    python -m dptab.cli composition   basic vs advanced vs RDP over many steps
    python -m dptab.cli calibrate     noise needed for a target epsilon, and what it costs
    python -m dptab.cli tradeoff      accuracy and attack success across the budget
    python -m dptab.cli groups        who pays for the privacy
    python -m dptab.cli all
"""

from __future__ import annotations

import random
import sys

from .audit import (
    audit_bound_is_consistent,
    canary_exposure,
    group_impact,
    membership_inference,
    utility_privacy_curve,
)
from .learning import make_dataset, train_dpsgd, train_sgd
from .mechanisms import (
    RDPAccountant,
    advanced_composition,
    basic_composition,
    clipped_mean,
    debias_randomised_response,
    gaussian_sigma,
    laplace_mechanism,
    noise_for_target_epsilon,
    private_histogram,
    randomised_response,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_mechanisms() -> None:
    rule("MECHANISMS -- noise is calibrated to sensitivity, not to taste")
    rng = random.Random(0)
    true_count = 4127
    print("a count (sensitivity 1), Laplace mechanism, 5 independent releases:")
    for epsilon in (0.1, 0.5, 1.0, 4.0):
        draws = [laplace_mechanism(true_count, 1.0, epsilon, rng) for _ in range(5)]
        spread = max(draws) - min(draws)
        print(
            f"  eps {epsilon:<5} " + "  ".join(f"{value:8.1f}" for value in draws) + f"   spread {spread:6.1f}"
        )

    print(f"\ntrue count {true_count}. Small epsilon is not a small effect on a small count.")

    print("\nthe same query on a subgroup of 40 rather than 4127:")
    for epsilon in (0.1, 1.0):
        draws = [laplace_mechanism(40, 1.0, epsilon, rng) for _ in range(5)]
        print(f"  eps {epsilon:<5} " + "  ".join(f"{value:8.1f}" for value in draws))
    print("Absolute noise is unchanged, so the relative error explodes. Small cells are where DP hurts.")

    print("\na mean with and without clipping (incomes, heavy tailed):")
    incomes = [rng.lognormvariate(10.5, 0.9) for _ in range(2000)]
    truth = sum(incomes) / len(incomes)
    for bound in (50_000.0, 150_000.0, 500_000.0):
        clipped_truth = sum(min(value, bound) for value in incomes) / len(incomes)
        private = clipped_mean(incomes, 0.0, bound, 1.0, rng)
        print(
            f"  clip at {bound:>9,.0f}: clipping bias {clipped_truth - truth:>+10.1f}, "
            f"private estimate {private:>10.1f} (truth {truth:.1f})"
        )
    print(
        "A tighter clip means less noise and more bias. On a heavy tail the bias dominates, and no\n"
        "amount of epsilon fixes it -- the estimator is answering a different question."
    )

    print("\nGaussian calibration (sensitivity 1, delta 1e-5):")
    for epsilon in (0.1, 0.5, 0.9):
        print(f"  eps {epsilon:<5} sigma {gaussian_sigma(1.0, epsilon, 1e-5):.2f}")
    try:
        gaussian_sigma(1.0, 2.0, 1e-5)
    except ValueError as error:
        print(f"  eps 2.0   refused: {error}")

    print("\nlocal DP by randomised response, true positive share 0.30:")
    for epsilon in (0.5, 1.0, 3.0):
        for n in (1_000, 100_000):
            answers = [randomised_response(rng.random() < 0.30, epsilon, rng) for _ in range(n)]
            estimate = debias_randomised_response(sum(answers) / n, epsilon)
            print(f"  eps {epsilon:<4} n {n:>7}: estimate {estimate:.4f}")
    print("Local DP needs enormous samples. That is the price of not trusting the curator.")

    print("\na histogram over public categories (sensitivity 1 per bucket, eps 1.0):")
    labels = ["a"] * 500 + ["b"] * 120 + ["c"] * 8 + ["d"] * 1
    noisy = private_histogram(labels, ("a", "b", "c", "d", "e"), 1.0, rng)
    for category, value in noisy.items():
        print(f"  {category}: {value:>8.1f}")
    print("The rare buckets are noise. Reported unclamped, because clamping biases exactly those.")


def demo_composition() -> None:
    rule("COMPOSITION -- the same 1,000 steps, three accountants")
    epsilon_step, delta_step, steps = 0.05, 0.0, 1000
    basic = basic_composition([epsilon_step] * steps, [delta_step] * steps)
    advanced = advanced_composition(epsilon_step, delta_step, steps, 1e-6)

    accountant = RDPAccountant()
    accountant.step(noise_multiplier=1.1, sample_rate=0.01, count=steps)
    rdp_epsilon, order = accountant.epsilon(1e-5)

    print(f"basic composition      eps = {basic[0]:8.2f}   (adds epsilons; always valid)")
    print(f"advanced composition   eps = {advanced[0]:8.2f}   (roughly sqrt(k) growth)")
    print(f"RDP, subsampled q=1%   eps = {rdp_epsilon:8.2f}   (order {order}; sigma 1.1)")
    print(
        "\nSubsampling is doing most of the work: a record not in the batch cannot have influenced\n"
        "the update. Without amplification, DP-SGD would be unusable at any interesting accuracy."
    )

    print("\nRDP epsilon growth with steps (sigma 1.1, q 1%, delta 1e-5):")
    for count in (100, 500, 1000, 5000, 20000):
        accountant = RDPAccountant()
        accountant.step(1.1, 0.01, count)
        print(f"  {count:>6} steps: eps {accountant.epsilon(1e-5)[0]:6.2f}")
    print("Roughly sqrt(k). Training longer is not free, and there is no schedule that makes it free.")


def demo_calibrate() -> None:
    rule("CALIBRATION -- pick the budget first, then find out what it costs")
    print("target eps   noise multiplier needed   (1,000 steps, q = 5%, delta = 1e-5)")
    for target in (0.5, 1.0, 2.0, 4.0, 8.0):
        sigma = noise_for_target_epsilon(target, 1e-5, 1000, 0.05)
        print(f"{target:>10}   {sigma:>23.3f}")
    print(
        "\nThe noise needed grows quickly as epsilon falls, and accuracy follows it down. Running the\n"
        "calculation this way round makes the budget a decision; the other way round makes it an\n"
        "accident that gets reported as a result."
    )


def demo_tradeoff() -> None:
    rule("THE TRADE -- accuracy, attack success, and the empirical bound")
    dataset = make_dataset(n=4000, canaries=20, seed=0)
    rows = utility_privacy_curve(dataset, epsilons=(0.5, 1.0, 2.0, 4.0, 8.0), seed=0)

    print(f"{'epsilon':>9} {'accuracy':>9} {'attack AUC':>11} {'advantage':>10} {'eps_lower':>10} {'clipped':>8}")
    for row in rows:
        label = "none" if row["epsilon"] is None else f"{row['epsilon']:.2f}"
        print(
            f"{label:>9} {row['accuracy']:>9.3f} {row['attack_auc']:>11.3f} "
            f"{row['advantage']:>10.3f} {row['empirical_epsilon']:>10.2f} {row['clip_rate']:>7.1%}"
        )

    train, holdout = dataset.split(0.7, seed=0)
    baseline = train_sgd(train, seed=0)
    private = train_dpsgd(train, target_epsilon=2.0, seed=0)
    print(f"\ncanary exposure, no privacy: {canary_exposure(baseline.model, train, holdout):+.4f}")
    print(f"canary exposure, eps = 2.0:  {canary_exposure(private.model, train, holdout):+.4f}")

    attack = membership_inference(private.model, train, holdout)
    consistent = audit_bound_is_consistent(attack, private.epsilon or 0.0)
    print(f"\n{attack.line()}")
    print(f"empirical lower bound within the accounted epsilon: {consistent}")
    print(
        "\nThat last line is the only privacy claim here that can fail. It cannot prove the guarantee --\n"
        "no finite experiment can -- but a violation would prove the implementation wrong."
    )


def demo_groups() -> None:
    rule("WHO PAYS -- disparate impact of DP-SGD")
    dataset = make_dataset(n=5000, minority_share=0.08, canaries=10, seed=1)
    train, holdout = dataset.split(0.7, seed=1)
    baseline = train_sgd(train, seed=1)

    for epsilon in (8.0, 2.0, 0.5):
        private = train_dpsgd(train, target_epsilon=epsilon, seed=1)
        impact = group_impact(baseline.model, private.model, holdout)
        print(f"\n--- eps = {private.epsilon:.2f}, clipped {private.clip_rate:.1%} of gradients ---")
        print(impact.report())

    print(
        "\nThe minority group follows a different decision rule and is 8% of the data, so its gradients\n"
        "stay large longest, get clipped hardest, and are then buried in noise. An aggregate accuracy\n"
        "figure hides this completely -- which is why the aggregate should not be the number reported."
    )


DEMOS = {
    "mechanisms": demo_mechanisms,
    "composition": demo_composition,
    "calibrate": demo_calibrate,
    "tradeoff": demo_tradeoff,
    "groups": demo_groups,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    choice = arguments[0] if arguments else "all"
    if choice == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if choice not in DEMOS:
        print(f"unknown demo {choice!r}\navailable: {', '.join(DEMOS)}, all")
        return 2
    DEMOS[choice]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
