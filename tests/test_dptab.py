"""Tests. Mechanism calibration, accountant identities, then DP-SGD's contract.

The load-bearing tests:

* ``test_subsampled_bound_reduces_to_the_gaussian_bound_at_full_sampling`` -- the one internal consistency
  check available on the Mironov formula. It caught a real bug in this repository.
* ``test_rdp_is_tighter_than_advanced_which_is_tighter_than_basic`` -- if the ordering ever inverts, the
  accountant is wrong, and a wrong accountant reports a comfortable epsilon for a run that has none.
* ``test_the_empirical_bound_never_exceeds_the_accounted_epsilon`` -- the falsification test. It cannot verify
  the guarantee; a failure would prove the implementation broken.
* ``test_noise_calibration_hits_its_target`` -- the calibration is a bisection, so it must be checked against
  the accountant it is inverting.

Mechanism randomness is checked statistically with fixed seeds and generous tolerances, since a test that fails
one run in twenty is worse than no test.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dptab.audit import (  # noqa: E402
    audit_bound_is_consistent,
    canary_exposure,
    group_impact,
    membership_inference,
)
from dptab.learning import (  # noqa: E402
    LogisticModel,
    clip_l2,
    make_dataset,
    train_dpsgd,
    train_sgd,
)
from dptab.mechanisms import (  # noqa: E402
    RDPAccountant,
    advanced_composition,
    basic_composition,
    clipped_mean,
    debias_randomised_response,
    gaussian_rdp,
    gaussian_sigma,
    laplace_mechanism,
    laplace_noise,
    noise_for_target_epsilon,
    private_histogram,
    randomised_response,
    rdp_to_dp,
    subsampled_gaussian_rdp,
)


class TestLaplace:
    def test_the_noise_has_the_right_scale(self):
        """Var[Lap(b)] = 2 b^2, so the standard deviation is b sqrt(2)."""
        rng = random.Random(0)
        draws = [laplace_noise(2.0, rng) for _ in range(40_000)]
        assert statistics.mean(draws) == pytest.approx(0.0, abs=0.05)
        assert statistics.stdev(draws) == pytest.approx(2.0 * math.sqrt(2.0), rel=0.05)

    def test_noise_scales_inversely_with_epsilon(self):
        rng = random.Random(1)
        loose = statistics.stdev([laplace_mechanism(0.0, 1.0, 2.0, rng) for _ in range(20_000)])
        tight = statistics.stdev([laplace_mechanism(0.0, 1.0, 0.2, rng) for _ in range(20_000)])
        assert tight == pytest.approx(10.0 * loose, rel=0.15)

    def test_the_mechanism_is_unbiased(self):
        rng = random.Random(2)
        draws = [laplace_mechanism(100.0, 1.0, 0.5, rng) for _ in range(20_000)]
        assert statistics.mean(draws) == pytest.approx(100.0, abs=0.5)

    def test_zero_sensitivity_and_zero_epsilon_are_rejected(self):
        rng = random.Random(0)
        with pytest.raises(ValueError, match="sensitivity"):
            laplace_mechanism(1.0, 0.0, 1.0, rng)
        with pytest.raises(ValueError, match="epsilon"):
            laplace_mechanism(1.0, 1.0, 0.0, rng)


class TestGaussianAndLocal:
    def test_sigma_matches_the_closed form_calibration(self):
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 0.5
        assert gaussian_sigma(1.0, 0.5, 1e-5) == pytest.approx(expected)

    def test_the_classical_bound_refuses_epsilon_above_one(self):
        """Not loose above eps = 1 -- wrong. Returning a number there would be the dangerous choice."""
        with pytest.raises(ValueError, match="requires eps < 1"):
            gaussian_sigma(1.0, 1.5, 1e-5)

    def test_more_privacy_needs_more_noise(self):
        assert gaussian_sigma(1.0, 0.1, 1e-5) > gaussian_sigma(1.0, 0.9, 1e-5)

    def test_randomised_response_debiases_to_the_truth(self):
        rng = random.Random(3)
        truth = 0.3
        answers = [randomised_response(rng.random() < truth, 1.0, rng) for _ in range(200_000)]
        estimate = debias_randomised_response(sum(answers) / len(answers), 1.0)
        assert estimate == pytest.approx(truth, abs=0.02)

    def test_local_dp_needs_far_more_samples_than_central(self):
        """The comparison that justifies the central model where a curator can be trusted."""
        rng = random.Random(4)
        truth = 0.3
        local = [
            abs(
                debias_randomised_response(
                    sum(randomised_response(rng.random() < truth, 1.0, rng) for _ in range(2000)) / 2000,
                    1.0,
                )
                - truth
            )
            for _ in range(20)
        ]
        central = [
            abs(laplace_mechanism(truth, 1.0 / 2000, 1.0, rng) - truth) for _ in range(20)
        ]
        assert statistics.mean(local) > 20 * statistics.mean(central)


class TestClippedMeanAndHistogram:
    def test_clipping_bounds_the_sensitivity_and_the_noise(self):
        rng = random.Random(5)
        values = [50.0] * 1000
        estimates = [clipped_mean(values, 0.0, 100.0, 1.0, rng) for _ in range(200)]
        assert statistics.mean(estimates) == pytest.approx(50.0, abs=0.5)

    def test_a_tight_clip_biases_a_heavy_tail(self):
        """The bias the noise cannot fix, and the reason a private mean needs its clip range justified."""
        rng = random.Random(6)
        values = [rng.lognormvariate(3.0, 1.2) for _ in range(4000)]
        truth = statistics.mean(values)
        tight = statistics.mean([clipped_mean(values, 0.0, 20.0, 4.0, rng) for _ in range(30)])
        wide = statistics.mean([clipped_mean(values, 0.0, 400.0, 4.0, rng) for _ in range(30)])
        assert tight < truth  # systematically low, not merely noisy
        assert abs(wide - truth) < abs(tight - truth)

    def test_an_inverted_clip_range_is_rejected(self):
        with pytest.raises(ValueError, match="upper must exceed lower"):
            clipped_mean([1.0], 5.0, 1.0, 1.0, random.Random(0))

    def test_the_histogram_is_unbiased_per_bucket(self):
        rng = random.Random(7)
        labels = ["a"] * 100 + ["b"] * 10
        totals = {"a": 0.0, "b": 0.0, "c": 0.0}
        for _ in range(400):
            noisy = private_histogram(labels, ("a", "b", "c"), 1.0, rng)
            for key, value in noisy.items():
                totals[key] += value
        assert totals["a"] / 400 == pytest.approx(100.0, abs=1.0)
        assert totals["b"] / 400 == pytest.approx(10.0, abs=1.0)
        assert totals["c"] / 400 == pytest.approx(0.0, abs=1.0)


class TestAccounting:
    def test_gaussian_rdp_matches_the_formula(self):
        assert gaussian_rdp(4.0, 2.0) == pytest.approx(4.0 / (2.0 * 4.0))

    def test_subsampled_bound_reduces_to_the_gaussian_bound_at_full_sampling(self):
        """The internal consistency check on Mironov's formula. It caught a real bug here."""
        for alpha in (2, 3, 8, 16):
            assert subsampled_gaussian_rdp(alpha, 1.5, 1.0) == pytest.approx(
                gaussian_rdp(alpha, 1.5), rel=1e-9
            )

    def test_subsampling_is_strictly_cheaper_than_full_batch(self):
        assert subsampled_gaussian_rdp(8, 1.1, 0.01) < subsampled_gaussian_rdp(8, 1.1, 1.0)

    def test_more_noise_costs_less_privacy(self):
        assert subsampled_gaussian_rdp(8, 2.0, 0.01) < subsampled_gaussian_rdp(8, 1.0, 0.01)

    def test_fractional_orders_are_refused_rather_than_rounded(self):
        with pytest.raises(ValueError, match="integer order"):
            subsampled_gaussian_rdp(2.5, 1.0, 0.01)  # type: ignore[arg-type]

    def test_rdp_composes_by_addition(self):
        """The property that makes RDP the right currency: k steps cost exactly k times one step."""
        one = RDPAccountant()
        one.step(1.1, 0.01, 1)
        many = RDPAccountant()
        many.step(1.1, 0.01, 500)
        for order in one.orders:
            assert many.spent[order] == pytest.approx(500 * one.spent[order], rel=1e-12)

    def test_rdp_is_tighter_than_advanced_which_is_tighter_than_basic(self):
        """If this ordering inverts, the accountant is reporting a comfortable epsilon for nothing."""
        steps, epsilon_step = 1000, 0.05
        basic = basic_composition([epsilon_step] * steps, [0.0] * steps)[0]
        advanced = advanced_composition(epsilon_step, 0.0, steps, 1e-6)[0]
        accountant = RDPAccountant()
        accountant.step(1.1, 0.01, steps)
        rdp = accountant.epsilon(1e-5)[0]
        assert rdp < advanced < basic

    def test_epsilon_grows_sublinearly_in_steps(self):
        first = RDPAccountant()
        first.step(1.1, 0.01, 1000)
        second = RDPAccountant()
        second.step(1.1, 0.01, 4000)
        # four times the steps costs less than four times the epsilon
        assert second.epsilon(1e-5)[0] < 4.0 * first.epsilon(1e-5)[0]

    def test_a_smaller_delta_costs_more_epsilon(self):
        accountant = RDPAccountant()
        accountant.step(1.1, 0.01, 500)
        assert accountant.epsilon(1e-7)[0] > accountant.epsilon(1e-3)[0]

    def test_the_conversion_matches_its_formula(self):
        assert rdp_to_dp(0.5, 8.0, 1e-5) == pytest.approx(0.5 + math.log(1e5) / 7.0)

    def test_noise_calibration_hits_its_target(self):
        for target in (0.5, 2.0, 8.0):
            sigma = noise_for_target_epsilon(target, 1e-5, 500, 0.05)
            accountant = RDPAccountant()
            accountant.step(sigma, 0.05, 500)
            assert accountant.epsilon(1e-5)[0] == pytest.approx(target, rel=0.02)

    def test_calibration_is_monotone_in_the_target(self):
        loose = noise_for_target_epsilon(8.0, 1e-5, 500, 0.05)
        tight = noise_for_target_epsilon(0.5, 1e-5, 500, 0.05)
        assert tight > loose

    def test_an_unreachable_target_raises(self):
        with pytest.raises(ValueError, match="cannot reach"):
            noise_for_target_epsilon(1e-9, 1e-5, 100_000, 1.0)


class TestClipping:
    def test_a_small_gradient_passes_through_untouched(self):
        gradient = [0.1, 0.2]
        clipped, was_clipped = clip_l2(gradient, 1.0)
        assert clipped == gradient
        assert was_clipped is False

    def test_a_large_gradient_is_scaled_to_the_bound_and_keeps_its_direction(self):
        clipped, was_clipped = clip_l2([3.0, 4.0], 1.0)
        assert was_clipped is True
        assert math.sqrt(sum(value**2 for value in clipped)) == pytest.approx(1.0)
        assert clipped[1] / clipped[0] == pytest.approx(4.0 / 3.0)

    def test_a_non_positive_bound_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            clip_l2([1.0], 0.0)


class TestLearning:
    def test_the_non_private_model_learns(self):
        dataset = make_dataset(n=2000, canaries=0, seed=0)
        train, holdout = dataset.split(0.7, seed=0)
        result = train_sgd(train, epochs=15, seed=0)
        assert result.model.accuracy(holdout) > 0.7
        assert result.epsilon is None  # no guarantee, and it says so

    def test_the_loss_curve_decreases(self):
        dataset = make_dataset(n=1500, canaries=0, seed=1)
        result = train_sgd(dataset, epochs=12, seed=1)
        assert result.loss_curve[-1] < result.loss_curve[0]

    def test_the_per_example_gradient_matches_a_finite_difference(self):
        """The gradient is written by hand, so it is checked numerically rather than trusted."""
        dataset = make_dataset(n=10, canaries=0, seed=2)
        record = dataset.records[0]
        model = LogisticModel([0.3, -0.2, 0.1, 0.4, -0.5], 0.2)
        analytic = model.gradient(record)
        step = 1e-6
        for index in range(len(model.weights)):
            shifted = LogisticModel(list(model.weights), model.bias)
            shifted.weights[index] += step
            numeric = (shifted.loss(record) - model.loss(record)) / step
            assert analytic[index] == pytest.approx(numeric, abs=1e-4)

    def test_dpsgd_reports_an_epsilon_and_learns_something(self):
        dataset = make_dataset(n=3000, canaries=0, seed=3)
        train, holdout = dataset.split(0.7, seed=3)
        result = train_dpsgd(train, target_epsilon=8.0, epochs=15, seed=3)
        assert result.epsilon == pytest.approx(8.0, rel=0.05)
        assert result.model.accuracy(holdout) > 0.6
        assert 0.0 <= result.clip_rate <= 1.0

    def test_a_tighter_budget_costs_accuracy(self):
        dataset = make_dataset(n=3000, canaries=0, seed=4)
        train, holdout = dataset.split(0.7, seed=4)
        loose = train_dpsgd(train, target_epsilon=8.0, epochs=15, seed=4)
        tight = train_dpsgd(train, target_epsilon=0.3, epochs=15, seed=4)
        assert loose.model.accuracy(holdout) >= tight.model.accuracy(holdout)

    def test_the_private_model_is_worse_than_the_baseline(self):
        """The price of the guarantee. A DP model that matched the baseline would mean a bug in the accounting."""
        dataset = make_dataset(n=3000, canaries=0, seed=5)
        train, holdout = dataset.split(0.7, seed=5)
        baseline = train_sgd(train, epochs=15, seed=5)
        private = train_dpsgd(train, target_epsilon=1.0, epochs=15, seed=5)
        assert private.model.accuracy(holdout) <= baseline.model.accuracy(holdout) + 1e-9

    def test_giving_neither_epsilon_nor_noise_is_an_error(self):
        dataset = make_dataset(n=200, canaries=0, seed=6)
        with pytest.raises(ValueError, match="either target_epsilon or noise_multiplier"):
            train_dpsgd(dataset, target_epsilon=None, noise_multiplier=None)


class TestAudit:
    def test_a_memorising_model_is_attackable_and_a_private_one_much_less_so(self):
        dataset = make_dataset(n=2500, canaries=40, seed=7)
        train, holdout = dataset.split(0.7, seed=7)
        baseline = train_sgd(train, epochs=40, learning_rate=0.6, seed=7)
        private = train_dpsgd(train, target_epsilon=1.0, epochs=20, seed=7)

        loud = membership_inference(baseline.model, train, holdout)
        quiet = membership_inference(private.model, train, holdout)
        assert loud.advantage >= quiet.advantage

    def test_dp_reduces_canary_exposure(self):
        dataset = make_dataset(n=2000, canaries=40, seed=8)
        train, holdout = dataset.split(0.7, seed=8)
        baseline = train_sgd(train, epochs=40, learning_rate=0.6, seed=8)
        private = train_dpsgd(train, target_epsilon=1.0, epochs=20, seed=8)
        assert canary_exposure(private.model, train, holdout) <= canary_exposure(
            baseline.model, train, holdout
        )

    def test_the_empirical_bound_never_exceeds_the_accounted_epsilon(self):
        """The falsification test: it cannot prove privacy, but a failure would prove a broken run."""
        dataset = make_dataset(n=2500, canaries=20, seed=9)
        train, holdout = dataset.split(0.7, seed=9)
        for target in (0.5, 2.0, 8.0):
            result = train_dpsgd(train, target_epsilon=target, epochs=15, seed=9)
            attack = membership_inference(result.model, train, holdout)
            assert audit_bound_is_consistent(attack, result.epsilon or 0.0), (
                target,
                attack.empirical_epsilon,
                result.epsilon,
            )

    def test_a_chance_level_attack_implies_no_lower_bound(self):
        from dptab.audit import AttackResult

        assert AttackResult(0.5, 0.0, 0.0, 0.5, 0.5).empirical_epsilon == pytest.approx(0.0)

    def test_group_impact_reports_a_per_group_cost(self):
        dataset = make_dataset(n=3000, minority_share=0.08, canaries=0, seed=10)
        train, holdout = dataset.split(0.7, seed=10)
        baseline = train_sgd(train, epochs=15, seed=10)
        private = train_dpsgd(train, target_epsilon=1.0, epochs=15, seed=10)
        impact = group_impact(baseline.model, private.model, holdout)
        assert set(impact.baseline) >= {"majority", "minority"}
        assert impact.disparity() >= 0.0

    def test_the_dataset_reports_its_own_composition(self):
        dataset = make_dataset(n=1000, minority_share=0.1, canaries=5, seed=11)
        counts = dataset.group_counts()
        assert counts["canary"] == 5
        assert counts["minority"] == 100
        assert sum(counts.values()) == 1005
