# Differential Privacy for Tabular ML

Laplace and Gaussian mechanisms, randomised response, RDP accounting for the **Poisson-subsampled Gaussian**,
DP-SGD with per-example gradient clipping, and a membership-inference audit that measures what the epsilon
actually buys. **Pure Python, standard library only** — no Opacus, no dp-accounting, no NumPy.

## Epsilon is a bound, not a result

```
P(M(D) in S) <= exp(eps) * P(M(D') in S) + delta        D, D' differ in one record
```

That is a promise about the mechanism, and it holds against any adversary with any side information — which is
why it survives post-processing and composition. What it does *not* say is how much a real attacker learns from
your model. The gap between the two is usually enormous, so this repository reports both: the accountant's
epsilon **and** an attack's empirical lower bound.

```bash
python -m dptab.cli mechanisms    # sensitivity, small cells, clipping bias
python -m dptab.cli composition   # basic vs advanced vs RDP over 1,000 steps
python -m dptab.cli calibrate     # noise needed for a target epsilon
python -m dptab.cli tradeoff      # accuracy, attack success, empirical epsilon
python -m dptab.cli groups        # who pays for the privacy
```

## Sensitivity is where DP systems actually fail

Noise is calibrated to how much one record can change the answer. Get that wrong and the noise is decoration:
the epsilon gets reported, and the guarantee is void.

- **A count** has sensitivity 1. Easy.
- **A sum or mean over an unbounded domain has unbounded sensitivity.** An average salary is not privately
  releasable until salaries are clipped, and *the clipping bound is then part of the privacy analysis*. Choosing
  it by looking at the data's min and max leaks the extremes — the most identifiable records in the dataset.
- **Clipping trades noise for bias**, and on a heavy tail the bias wins:

```
clip at    50,000: clipping bias   -18,204.3
clip at   150,000: clipping bias    -4,881.7
clip at   500,000: clipping bias      -712.4
```

No epsilon repairs that. A private mean income with a 50k clip is not an estimate of mean income.

- **Small cells are where DP hurts.** The same query, absolute noise unchanged: on 4,127 records the noise is
  irrelevant; on a subgroup of 40 it is the answer. Every "DP made our dashboard useless" story is this.
- **Histogram buckets must be public.** Deriving categories from the data leaks a record through the mere
  *presence* of a bucket — the classic case where a noisy count of a rare diagnosis still reveals that someone
  has it.

## Composition: the same 1,000 steps, three accountants

```
basic composition      eps =    50.00     (adds epsilons; always valid)
advanced composition   eps =     8.83     (roughly sqrt(k) growth)
RDP, subsampled q=1%   eps =     1.31     (order 16, sigma 1.1)
```

RDP composes **exactly by addition** at every order — the tests assert 500 steps cost precisely 500 times one
step — and converts to `(eps, delta)` once at the end, taking the best order. Most of the remaining gain comes
from **amplification by subsampling**: a record not in the batch cannot have influenced the update. Without it
DP-SGD would be unusable at any interesting accuracy.

One caveat worth stating: the conversion used here is the standard loose one. Tighter conversions give a
visibly smaller epsilon for the *same run*, so a headline epsilon depends on the accountant as much as on the
noise — which matters when comparing numbers across papers.

## DP-SGD, and the two details that quietly remove the guarantee

```
g_i     = clip(grad_i, C)                          per-example, not batch
g_noisy = (sum_i g_i + N(0, (sigma C)^2 I)) / E[B]
w      <- w - lr * g_noisy
```

1. **Poisson sampling, not shuffling.** The amplification bound assumes each record is included independently
   with probability `q`. Fixed-size shuffled minibatches — the default in every framework — do not satisfy it,
   and the epsilon you report is then understated. Batch sizes here therefore vary.
2. **Divide by the expected batch size, not the realised one.** The realised count is a data-dependent quantity;
   dividing by it leaks. A one-line difference between DP and DP-flavoured.

Noise is added to the *sum*, so per-coordinate noise falls as the batch grows: DP-SGD wants **large** batches,
the opposite of the usual advice. The returned `clip_rate` is the diagnostic that matters — near 100% means the
update direction is being set by clipping rather than by data.

## What the epsilon buys, measured

```
  epsilon  accuracy  attack AUC  advantage  eps_lower  clipped
     none     0.812       0.607      0.148       0.61    0.0%
     0.50     0.703       0.502      0.011       0.04   99.2%
     1.00     0.729       0.505      0.018       0.07   96.8%
     2.00     0.751       0.509      0.024       0.09   91.4%
     8.00     0.784       0.523      0.041       0.16   74.1%
```

> Illustrative and seed-dependent; the tests assert the orderings, not the digits.

Read the last column with the first: at `eps = 0.5` almost every gradient is clipped, so the model is barely
learning from the data at all. And note the **empirical epsilon is far below the accounted one** everywhere.
That is the normal situation, and it cuts both ways:

- a successful attack **proves** a leak;
- a failed attack proves only that *this* attack failed. Shadow-model attacks (LiRA) do considerably better, so
  the empirical bound is a floor on leakage, never a ceiling.

The one privacy claim here that can fail is `audit_bound_is_consistent`: if a measured lower bound ever exceeded
the accountant's epsilon, the implementation would be provably broken. No finite experiment can verify a DP
guarantee — but it can refute one, and the failures it catches (mis-stated sensitivity, a sampling scheme that
does not match the amplification bound) are otherwise completely silent.

## Privacy is not paid for equally

```
group        baseline   private     cost
canary          0.900     0.500    0.400
majority        0.831     0.792    0.039
minority        0.724     0.531    0.193

disparity between best- and worst-hit group: 0.361
```

The minority group is 8% of the data and follows a different decision rule, so its gradients stay large longest,
get clipped hardest, and are then buried in noise. An aggregate accuracy figure hides this entirely — which is
why the aggregate should not be the number reported. The canary row is the same mechanism working as intended:
those records are *supposed* to become unlearnable.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Laplace scale checked statistically against `b*sqrt(2)`, the Gaussian calibration refusing `eps >= 1`,
randomised response debiasing to the truth, local DP needing 20x the error of central DP at matched epsilon,
clipping bias on a lognormal, Mironov's bound reducing to the plain Gaussian bound at `q = 1`, exact RDP
additivity, the `RDP < advanced < basic` ordering, the calibration bisection agreeing with the accountant it
inverts, the hand-written gradient against a finite difference, and the empirical epsilon staying inside the
accounted epsilon at three budgets.

## Limits

- **Logistic regression only.** No deep networks; the accounting is identical but per-example clipping in a
  deep model needs microbatching or per-sample gradient machinery, and the utility story is much worse.
- **The loose RDP conversion.** Tighter conversions (Canonne-Kamath-Steinke, the analytic Gaussian mechanism)
  report smaller epsilon for the same run. Safe in direction, and a real difference in practice.
- **No hyperparameter privacy.** Choosing the clip bound, learning rate and epochs by looking at results is
  itself a data-dependent computation that consumes budget. Nobody accounts for it — including this repository —
  and it is the largest unacknowledged leak in most published DP results.
- **Record-level, not user-level.** One person contributing many rows gets much weaker protection than the
  epsilon suggests. Group privacy scales epsilon by the number of rows.
- **A weak attack.** Loss thresholding, not shadow models or LiRA, so the empirical bound is conservative.
- **Synthetic data**, and the RNG is `random`, not a cryptographic source — fine for demonstration, not for a
  deployment, where floating-point noise sampling has its own documented attacks (Mironov, 2012).

## References

- Dwork & Roth (2014), *The Algorithmic Foundations of Differential Privacy*.
- Abadi et al. (2016), *Deep learning with differential privacy* — DP-SGD and the moments accountant.
- Mironov (2017), *Rényi differential privacy*; Mironov, Talwar & Zhang (2019), *R'enyi DP of the sampled Gaussian mechanism*.
- Balle & Wang (2018), *Improving the Gaussian mechanism via optimal variance* — the analytic mechanism.
- Shokri et al. (2017), *Membership inference attacks against machine learning models*.
- Yeom et al. (2018), *Privacy risk in machine learning*; Carlini et al. (2022), *Membership inference attacks from first principles* (LiRA).
- Jagielski, Ullman & Oprea (2020), *Auditing differentially private machine learning*.
- Bagdasaryan, Poursaeed & Shmatikov (2019), *Differential privacy has disparate impact on model accuracy*.
- Mironov (2012), *On significance of the least significant bits for differential privacy*.

MIT licensed.
