"""CPCV — Combinatorial Purged Cross-Validation (Lopez de Prado AFML Ch.12 / MLAM Ch.7).

Multi-path robustness for a CONTINUOUS edge. The Shape Validator's WFO (test 1) gives ONE
chronological train/test path — the edge can survive the single split by luck. CPCV evaluates the
edge over MANY combinatorial sub-period paths and reports the DISTRIBUTION: a robust edge stays
profitable on (almost) every path; a time-concentrated / lucky edge collapses when a few periods
are removed. This is the combinatorial generalization of WFO, and it complements:
  - block bootstrap (test 2): WITH-replacement resample -> CI on the mean.
  - PurgedKFold (test 5): 5 folds, positive-fold count.
  - CSCV (test 6): cohort-level PBO (selection overfitting across variants).
  CPCV is the leave-k-TIME-GROUP-out view for CONTINUOUS edges (lumpy, episodic payoffs need a
  leave-one-episode-out test instead).

SCOPING (honest): pure AFML CPCV reconstructs phi = C(N,k)*k/N backtest PATHS by training a model on
the (N-k) train groups and predicting the k test groups. For my RULE-BASED gates there is NOTHING
to fit, so the path reconstruction DEGENERATES (every path = the full realized series). The
applicable, additive statistic for a model-free daily-return strategy is the LEAVE-k-GROUP-OUT
Sharpe distribution with purge+embargo: enumerate the C(N,k) ways to hold out k of N contiguous
time groups, embargo the observations adjacent to each removed block, and Sharpe the retained
remainder. That is `cpcv_subperiod_sharpes` below (the deliverable). The fit/predict path engine is
DEFERRED until there is a fitted model to validate — no consumer today, so not built (YAGNI); the group/embargo helpers here are the reusable core.

Deterministic; no annualization assumptions leak into the verdict (Sharpe ratios compared like-for-
like across paths). Pure numpy.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CPCVResult:
    n_groups: int
    n_test: int                 # k = groups held out per path
    n_paths: int                # C(n_groups, k)
    embargo_obs: int
    full_sharpe: float          # Sharpe on the whole series (reference)
    sharpes: np.ndarray         # one Sharpe per combinatorial path
    median: float
    mean: float
    frac_positive: float        # fraction of paths with Sharpe > 0
    p05: float                  # 5th-percentile path Sharpe (worst-decile-ish)
    p95: float
    worst: float
    verdict: str = "n/a"        # ROBUST | MODERATE | CONCENTRATED


def _make_groups(T: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition [0, T) into n_groups CONTIGUOUS, ~even index ranges."""
    if n_groups < 2 or T < n_groups * 2:
        raise ValueError(f"need T>=2*n_groups; got T={T}, n_groups={n_groups}")
    edges = np.linspace(0, T, n_groups + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def _retained_mask(T: int, groups: list[tuple[int, int]], held_out: tuple[int, ...],
                   embargo_obs: int) -> np.ndarray:
    """Boolean mask of the RETAINED observations: all groups minus the held-out groups, minus
    `embargo_obs` observations immediately before and after each held-out block (purge+embargo at
    the seams — prevents serial-correlation leakage across the cut, AFML Ch.7/12)."""
    mask = np.ones(T, dtype=bool)
    for gi in held_out:
        s, e = groups[gi]
        mask[s:e] = False
        lo = max(0, s - embargo_obs); mask[lo:s] = False           # embargo before
        hi = min(T, e + embargo_obs); mask[e:hi] = False           # embargo after
    return mask


def _sharpe(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def cpcv_subperiod_sharpes(returns, n_groups: int = 8, n_test: int = 2,
                           embargo_pct: float = 0.01) -> CPCVResult:
    """Leave-k-group-out combinatorial purged Sharpe distribution for a model-free daily-return
    strategy. Holds out every C(n_groups, n_test) combination of contiguous time-groups (with an
    embargo at the seams) and Sharpes the retained remainder.

    returns      : 1D array-like of per-period strategy returns (e.g. daily PnL).
    n_groups (N) : number of contiguous time groups.
    n_test  (k)  : groups held out per path -> C(N,k) paths.
    embargo_pct  : embargo as a fraction of T, applied on each side of each held-out block.

    Returns CPCVResult. Sharpe is the bare mean/std ratio (annualization cancels in cross-path
    comparison); the VERDICT keys off the path distribution, not the level.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < n_groups * 2:                       # too few observations -> empty (safe) result
        return CPCVResult(n_groups, n_test, 0, 0, float("nan"), np.array([]),
                          float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"), "n/a")
    groups = _make_groups(T, n_groups)
    embargo_obs = int(round(embargo_pct * T))
    sharpes = []
    for held in itertools.combinations(range(n_groups), n_test):
        m = _retained_mask(T, groups, held, embargo_obs)
        s = _sharpe(r[m])
        if math.isfinite(s):
            sharpes.append(s)
    arr = np.array(sharpes, dtype=float)
    full = _sharpe(r)
    if len(arr) == 0:
        return CPCVResult(n_groups, n_test, 0, embargo_obs, full, arr,
                          float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"), "n/a")
    frac_pos = float((arr > 0).mean())
    p05 = float(np.percentile(arr, 5)); p95 = float(np.percentile(arr, 95))
    verdict = ("ROBUST" if (frac_pos >= 0.95 and p05 > 0)
               else "MODERATE" if frac_pos >= 0.80
               else "CONCENTRATED")
    return CPCVResult(
        n_groups=n_groups, n_test=n_test, n_paths=len(arr), embargo_obs=embargo_obs,
        full_sharpe=full, sharpes=arr, median=float(np.median(arr)), mean=float(arr.mean()),
        frac_positive=frac_pos, p05=p05, p95=p95, worst=float(arr.min()), verdict=verdict,
    )


def cpcv_pass(res: CPCVResult, min_frac_positive: float = 0.95, require_p05_positive: bool = True) -> bool:
    """ROBUSTNESS gate: the edge holds on (almost) every combinatorial sub-period path. A
    CONCENTRATED verdict means the edge depends on a few time-groups -> route to the episodic
    validator (lumpy-payoff class, codex P1) rather than treating it as a continuous edge."""
    if res.n_paths == 0 or not math.isfinite(res.frac_positive):
        return False
    if require_p05_positive and not (res.p05 > 0):
        return False
    return res.frac_positive >= min_frac_positive


def cpcv_report(res: CPCVResult) -> str:
    return (f"CPCV multi-path (N={res.n_groups}, k={res.n_test}, {res.n_paths} paths, "
            f"embargo {res.embargo_obs}obs): full Sh {res.full_sharpe:+.2f} | "
            f"paths median {res.median:+.2f} p05 {res.p05:+.2f} worst {res.worst:+.2f} | "
            f"{100*res.frac_positive:.0f}% paths>0 -> {res.verdict}")


# ---------------------------------------------------------------- self-test (known answers)
def _selftest() -> None:
    rng = np.random.default_rng(12)
    T = 1200
    # 1. ROBUST continuous edge: steady positive drift -> almost every sub-period Sharpe > 0
    good = rng.normal(0.0010, 0.010, T)
    rg = cpcv_subperiod_sharpes(good, n_groups=8, n_test=2)
    assert rg.verdict == "ROBUST" and rg.frac_positive >= 0.95, f"robust fail: {cpcv_report(rg)}"
    # 2. LUMPY edge: the rest of the tape slightly LOSES and one fat group carries the whole edge;
    #    full Sharpe is positive but the paths that hold out the fat group go negative (the
    #    discriminator the single full-sample Sharpe hides).
    conc = rng.normal(-0.0006, 0.010, T)
    conc[150:300] += 0.020                                   # one fat group carries everything
    rc = cpcv_subperiod_sharpes(conc, n_groups=8, n_test=2)
    assert rc.full_sharpe > 0 and rc.verdict != "ROBUST" and rc.frac_positive < rg.frac_positive, \
        f"lumpy fail: {cpcv_report(rc)}"
    # 3. NOISE: ~half the paths positive, not ROBUST
    noise = rng.normal(0.0, 0.010, T)
    rn = cpcv_subperiod_sharpes(noise, n_groups=8, n_test=2)
    assert rn.verdict != "ROBUST", f"noise fail: {cpcv_report(rn)}"
    assert not cpcv_pass(rn) and cpcv_pass(rg), "cpcv_pass gate fail"
    print("[self-test] PASS")
    print("  robust      :", cpcv_report(rg))
    print("  concentrated:", cpcv_report(rc))
    print("  noise       :", cpcv_report(rn))


if __name__ == "__main__":
    _selftest()
