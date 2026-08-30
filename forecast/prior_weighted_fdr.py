"""How good would Sonnet's priors have to BE for prior-weighted FDR to help?

The proposal: Sonnet assigns each condition a plausibility weight at
PROPOSAL TIME, locked before any result is seen, and Benjamini-Hochberg
allocates alpha in proportion. A well-motivated hypothesis gets a larger
share of the error budget, a scattershot one less. FDR control holds for
any fixed weights, so bad priors cost power but cannot manufacture false
discoveries.

That is the theory. Whether it is worth paying Sonnet for is an empirical
question about ONE number: how strongly its judgement correlates with
which hypotheses are actually real. This measures that, offline and free,
before a single API call is spent.

Why this cannot be measured on the existing control sweep. The obvious
test -- weight the planted arms up and the noise arm down -- is circular:
the arm label IS the ground truth, so any "prior" derived from it is a
perfect oracle by construction and would prove only that oracles work.
Simulation is the honest instrument here, because the prior's quality can
be set as a parameter and swept.

Model, matched to this project's measured regime:
  * family size m ~ 300 (the actionable set from the real grammar sweep)
  * a small fraction of hypotheses are genuinely non-null
  * non-null p-values are drawn at the LOW POWER actually measured here,
    not textbook power -- weighting cannot rescue a test that never
    detects anything, and assuming otherwise would flatter the result
  * prior quality `q` is the correlation between Sonnet's plausibility
    score and the truth: q=0 is noise, q=1 is an oracle

Reading it: compare true discoveries at each q against the unweighted
baseline, and check realised FDR stays at or below alpha throughout. If
realistic prior quality buys little, the honest answer is to skip it and
keep the pipeline simpler.

Run:  python3 -m forecast.prior_weighted_fdr
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

M = 300              # family size, from the real grammar sweep's actionable set
PI_TRUE = 0.05       # fraction genuinely non-null -- deliberately pessimistic
POWER = 0.27         # measured detection rate at alpha=0.10 on planted signals
ALPHA = 0.05
TRIALS = 4000
QS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _p_values(rng, n_true: int, n_null: int):
    """Null p-values are Uniform(0,1). Non-null ones are drawn so that the
    fraction below alpha matches the POWER this project actually has --
    a Beta skewed toward zero, rather than the near-zero p-values a
    well-powered study would produce."""
    null = rng.uniform(size=n_null)
    # Beta(a,1) puts mass near 0; pick `a` so P(p < ALPHA) ~= POWER.
    a = np.log(POWER) / np.log(ALPHA) if 0 < POWER < 1 else 1.0
    true = rng.beta(a, 1.0, size=n_true)
    return true, null


def _weights(rng, truth: np.ndarray, q: float) -> np.ndarray:
    """A plausibility score correlated `q` with the truth, mapped to
    positive weights. Assigned WITHOUT reference to the p-values -- which
    is the whole point, and the one condition the guarantee depends on."""
    z = (truth - truth.mean()) / (truth.std() or 1.0)
    score = q * z + np.sqrt(max(1 - q * q, 0.0)) * rng.standard_normal(len(truth))
    return np.exp(score)          # positive, and normalised inside benjamini_hochberg


def main() -> None:
    from candidates.methodology import benjamini_hochberg

    rng = np.random.default_rng(20260830)
    n_true = int(M * PI_TRUE)
    n_null = M - n_true
    truth = np.array([1] * n_true + [0] * n_null)

    print(f"Prior-weighted FDR: is it worth asking Sonnet for plausibility weights?\n")
    print(f"  family m={M}, genuinely non-null={n_true} ({PI_TRUE:.0%}), "
          f"power={POWER:.0%}, alpha={ALPHA}, {TRIALS} trials\n")
    print(f"{'prior quality q':>16}{'true found':>13}{'vs unweighted':>15}{'realised FDR':>14}")
    print("-" * 60)

    base_true = None
    for q in QS:
        found_true = 0
        # FDR is E[V / max(R,1)] -- the EXPECTATION OF THE PER-TRIAL RATIO, not
        # the ratio of the pooled totals. The two differ sharply when most
        # trials make zero discoveries, which is exactly this regime: pooling
        # reported 16.4% at the unweighted baseline and made a correctly
        # behaving BH look like it was failing to control FDR at all.
        fdp = []
        for _ in range(TRIALS):
            tp, np_ = _p_values(rng, n_true, n_null)
            ps = np.concatenate([tp, np_])
            w = _weights(rng, truth, q) if q > 0 else None
            keep = benjamini_hochberg(list(ps), ALPHA, list(w) if w is not None else None)
            k = np.array(keep)
            v = int((k & (truth == 0)).sum())
            r = int(k.sum())
            found_true += int((k & (truth == 1)).sum())
            fdp.append(v / r if r else 0.0)
        avg_true = found_true / TRIALS
        fdr = float(np.mean(fdp))
        if base_true is None:
            base_true = avg_true
        delta = (avg_true / base_true - 1) * 100 if base_true else 0.0
        print(f"{q:>16.1f}{avg_true:>13.2f}{delta:>+14.0f}%{fdr:>14.1%}")

    print("\n" + "=" * 60)
    print("q=0.0 is the unweighted baseline (noise priors). Realised FDR must stay")
    print(f"at or under alpha={ALPHA} at EVERY q -- that is the property that makes")
    print("bad priors merely wasteful rather than dangerous.")


if __name__ == "__main__":
    sys.exit(main())
