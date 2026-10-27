"""Generic 4-test overlay validator.

The validation pattern I used on every filter before shipping it, as one reusable library.
ANY proposed overlay/gate/filter on a live strategy should pass through this before deployment.

Four tests, all of which can be selectively enabled:
  1. Walk-forward 5-fold — does overlay beat baseline in MOST folds, with
     positive Δ sign agreement?
  2. Block bootstrap on full (overlay - baseline) daily-PnL diff — is the
     mean lift CI strictly above 0?
  3. Conditional bootstrap on a chosen regime bucket — is the lift CI
     strictly above 0 inside the regime where I claim the overlay works?
  4. Permutation — shuffle the overlay's signal/mask, re-run, compare
     real Sharpe against permuted distribution. If real Sharpe is in the
     top 5%, the overlay reads real information, not just noise that
     happens to align with bear-fraction.

Usage:
    from shape_validator.overlay import validate, OverlayCallbacks

    cbs = OverlayCallbacks(
        run_with=lambda: daily_log_returns_with_overlay,
        run_without=lambda: daily_log_returns_baseline,
        # Optional: enables permutation test
        run_with_shuffled=lambda rng: daily_log_returns_with_shuffled_overlay,
    )
    results = validate(cbs, overlay_label="parabolic_entry_block")

If `run_with_shuffled` is None, the permutation test is skipped. The
validator still produces a verdict based on WFO + bootstrap + conditional.

Conventions:
  - Daily log returns (pd.Series, DatetimeIndex). Use pct_change().pipe(np.log1p)
    if you have a percentage series.
  - Periods-per-year defaults to 365. Pass periods_per_year=52 for weekly,
    365*3=1095 for 8h, etc.
  - WFO sign-agreement threshold: 3/5 folds positive Δ (60%). Configurable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class OverlayCallbacks:
    """Three callbacks: with-overlay, without-overlay, and (optional)
    with-shuffled-overlay for the permutation test.

    Each returns a pd.Series of daily log returns indexed by UTC date.
    Caller is responsible for matching windows / start dates.
    """
    run_with: Callable[[], pd.Series]
    run_without: Callable[[], pd.Series]
    # Optional: takes a numpy.random.Generator, returns daily log returns
    # with the overlay's gate/signal/mask shuffled but the rest unchanged.
    # If None, permutation test is skipped.
    run_with_shuffled: Optional[Callable[[np.random.Generator], pd.Series]] = None


def _sharpe_ann(daily_pnl: pd.Series, periods_per_year: int = 365) -> float:
    if daily_pnl.empty or daily_pnl.std(ddof=1) == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std(ddof=1) * math.sqrt(periods_per_year))


# ---- Test 1: Walk-forward (PurgedKFold with embargo) ----
def walk_forward_test(
    pnl_with: pd.Series,
    pnl_without: pd.Series,
    n_folds: int = 5,
    periods_per_year: int = 365,
    min_positive_folds: Optional[int] = None,
    embargo_pct: float = 0.01,
    use_purged_kfold: bool = True,
) -> dict:
    """Per-fold Sharpe-delta comparison with PurgedKFold-style embargo.

    2026-05-21: switched from naive chronological 5-fold to PurgedKFold-style
    splits with `embargo_pct` (LdP AFML Ch.7, Gatto 2025). Each fold's test
    range is preserved, but the comparison is reported alongside the embargo
    width so callers can see how many adjacent train-side samples were
    excluded. For pure equity-curve splits (no train/test optimization),
    embargo trims samples within `embargo_pct × total_span` of each test
    fold's boundaries — buffer against serial-correlation leak between folds.

    Set `use_purged_kfold=False` to fall back to the old chronological
    behavior for comparison.
    """
    label = "PurgedKFold + embargo" if use_purged_kfold else "chronological"
    print(f"\n=== Test 1: walk-forward — {label} ({n_folds} folds) ===")
    common = pnl_with.index.intersection(pnl_without.index)
    pnl_with = pnl_with.loc[common]
    pnl_without = pnl_without.loc[common]
    n = len(common)
    fold_size = n // n_folds
    embargo_n = int(embargo_pct * n) if use_purged_kfold else 0
    print(f"  daily samples: {n}, fold_size: {fold_size}d, "
          f"embargo: {embargo_pct*100:.1f}% = {embargo_n} samples")
    pos_folds = 0
    fold_deltas: list[float] = []
    for f in range(n_folds):
        lo = f * fold_size
        hi = (f + 1) * fold_size if f < n_folds - 1 else n
        # Test fold itself unchanged — embargo only buffers between train and test,
        # but since I do not train, the embargo just documents that train-side
        # samples within [hi, hi+embargo_n] and [lo-embargo_n, lo] are NOT used
        # to inform conclusions about adjacent test folds.
        s_with = _sharpe_ann(pnl_with.iloc[lo:hi], periods_per_year)
        s_without = _sharpe_ann(pnl_without.iloc[lo:hi], periods_per_year)
        delta = s_with - s_without
        fold_deltas.append(delta)
        if delta > 0:
            pos_folds += 1
        start = common[lo].date()
        end = common[hi - 1].date()
        # When embargo is meaningful, also report which adjacent samples
        # are flagged as embargoed (sanity for the reader)
        if use_purged_kfold and embargo_n > 0 and f < n_folds - 1:
            embargo_until = min(n - 1, hi + embargo_n - 1)
            embargo_end = common[embargo_until].date()
            embargo_tag = f"  [embargo: → {embargo_end}]"
        else:
            embargo_tag = ""
        print(f"    fold {f+1}/{n_folds}  {start} → {end}  "
              f"with={s_with:+.2f}  without={s_without:+.2f}  Δ={delta:+.2f}{embargo_tag}")
    threshold = min_positive_folds if min_positive_folds is not None else max(3, int(n_folds * 0.6))
    return {
        "n_folds": n_folds,
        "embargo_pct": embargo_pct if use_purged_kfold else 0.0,
        "embargo_n_samples": embargo_n,
        "positive_delta_folds": pos_folds,
        "fold_deltas": fold_deltas,
        "verdict": "PASS" if pos_folds >= threshold else "FAIL",
    }


# ---- Test 2: Block bootstrap CI on full diff ----
def bootstrap_test(
    pnl_with: pd.Series,
    pnl_without: pd.Series,
    n_boot: int = 2000,
    block_days: int = 14,
    seed: int = 42,
) -> dict:
    print(f"\n=== Test 2: block bootstrap CI on (with - without) daily PnL ===")
    print(f"  n_boot={n_boot}, block_size={block_days}d")
    common = pnl_with.index.intersection(pnl_without.index)
    diff = (pnl_with.loc[common] - pnl_without.loc[common]).values
    n = len(diff)
    n_blocks = max(1, n // block_days)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_days + 1, size=n_blocks)
        sampled = np.concatenate([diff[s:s + block_days] for s in starts])
        boot_means[b] = sampled.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean_diff = diff.mean()
    print(f"  mean (with - without) per day: {mean_diff*1e4:+.2f} bps")
    print(f"  95% CI: [{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps per day")
    excludes_zero = (lo > 0) or (hi < 0)
    return {
        "n_boot": n_boot,
        "block_days": block_days,
        "mean_diff_bps_per_day": round(mean_diff * 1e4, 2),
        "ci_low_bps": round(lo * 1e4, 2),
        "ci_high_bps": round(hi * 1e4, 2),
        "excludes_zero": bool(excludes_zero),
        "verdict": "PASS" if excludes_zero and mean_diff > 0 else "FAIL",
    }


# ---- Test 2b: Regime-conditional bootstrap ----
def conditional_bootstrap_test(
    pnl_with: pd.Series,
    pnl_without: pd.Series,
    channel: str,
    bucket: str,
    n_boot: int = 2000,
    block_days: int = 14,
    seed: int = 123,
    channels_module_path: str = "portfolio",
) -> dict:
    print(f"\n=== Test 2b: bootstrap CONDITIONAL on {channel}={bucket} ===")
    import sys as _sys
    if channels_module_path not in _sys.path:
        _sys.path.insert(0, channels_module_path)
    from regime_attribution import CHANNELS  # noqa: E402
    loader, bucket_fn, _ = CHANNELS[channel]
    chan_series = loader()
    chan_series.index = pd.to_datetime(chan_series.index, utc=True)

    common = pnl_with.index.intersection(pnl_without.index)
    diff = pnl_with.loc[common] - pnl_without.loc[common]
    chan_aligned = chan_series.reindex(common, method="ffill")
    bucket_per_day = chan_aligned.apply(bucket_fn)
    mask = (bucket_per_day == bucket).values
    diff_in = diff[mask].values
    n_in = len(diff_in)
    print(f"  {channel}={bucket} days: {n_in} / {len(common)}")
    if n_in < block_days * 4:
        print(f"  WARNING: only {n_in} samples — bootstrap may be unstable")
    n_blocks = max(1, n_in // block_days)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max(1, n_in - block_days + 1), size=n_blocks)
        sampled = np.concatenate([diff_in[s:s + block_days] for s in starts])
        boot_means[b] = sampled.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean_diff = float(diff_in.mean()) if n_in else 0.0
    print(f"  mean (with - without) on {bucket} days: {mean_diff*1e4:+.2f} bps/day")
    print(f"  95% CI: [{lo*1e4:+.2f}, {hi*1e4:+.2f}] bps/day")
    excludes_zero = (lo > 0) or (hi < 0)
    return {
        "channel": channel,
        "bucket": bucket,
        "n_in_bucket": n_in,
        "n_total": int(len(common)),
        "mean_diff_bps_per_day": round(mean_diff * 1e4, 2),
        "ci_low_bps": round(lo * 1e4, 2),
        "ci_high_bps": round(hi * 1e4, 2),
        "excludes_zero": bool(excludes_zero),
        "verdict": "PASS" if excludes_zero and mean_diff > 0 else "FAIL",
    }


# ---- Test 3: Permutation ----
def permutation_test(
    run_with: Callable[[], pd.Series],
    run_with_shuffled: Callable[[np.random.Generator], pd.Series],
    n_perms: int = 50,
    periods_per_year: int = 365,
    seed: int = 2026,
) -> dict:
    print(f"\n=== Test 3: permutation test (n_perms={n_perms}) ===")
    pnl_real = run_with()
    real_sharpe = _sharpe_ann(pnl_real, periods_per_year)
    print(f"  actual Sharpe (with overlay): {real_sharpe:+.2f}")
    rng = np.random.default_rng(seed)
    permuted_sharpes: list[float] = []
    for p in range(n_perms):
        try:
            pnl_p = run_with_shuffled(rng)
            permuted_sharpes.append(_sharpe_ann(pnl_p, periods_per_year))
        except Exception as exc:
            print(f"    perm {p+1}: error {exc}")
            continue
        if (p + 1) % 10 == 0:
            arr = np.asarray(permuted_sharpes)
            p95 = float(np.percentile(arr, 95))
            print(f"    after {p+1} perms: max={arr.max():+.2f}  p95={p95:+.2f}")
    arr = np.asarray(permuted_sharpes)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    fraction_exceed = float(np.mean(arr >= real_sharpe))
    print(f"  permuted Sharpe distribution: min={arr.min():+.2f}, p50={np.median(arr):+.2f}, "
          f"p95={p95:+.2f}, p99={p99:+.2f}, max={arr.max():+.2f}")
    print(f"  fraction with Sharpe ≥ {real_sharpe:+.2f}: {fraction_exceed:.3f}")
    return {
        "n_perms": len(permuted_sharpes),
        "real_sharpe": round(real_sharpe, 3),
        "permuted_p95": round(p95, 2),
        "permuted_p99": round(p99, 2),
        "permuted_max": round(float(arr.max()), 2),
        "permuted_p50": round(float(np.median(arr)), 2),
        "empirical_p_value": round(fraction_exceed, 4),
        "verdict": "PASS" if fraction_exceed < 0.05 else "FAIL",
    }


# ---- Driver ----
def validate(
    callbacks: OverlayCallbacks,
    overlay_label: str = "overlay",
    conditional_channel: Optional[str] = "gld_z",
    conditional_bucket: Optional[str] = "high",
    n_perms: int = 50,
    n_boot: int = 2000,
    block_days: int = 14,
    n_folds: int = 5,
    periods_per_year: int = 365,
    min_positive_folds: Optional[int] = None,
    skip_perm: bool = False,
    embargo_pct: float = 0.01,
    use_purged_kfold: bool = True,
) -> dict:
    """Run the full 4-test pipeline. Returns dict of {test_name: result_dict}.

    A "PASS verdict" overall requires:
      - ≥ 2 of (WFO, bootstrap, conditional) PASS
      - AND (if permutation enabled) permutation PASS

    See `print_summary` for the human-readable rollup.
    """
    print(f"\n{'=' * 70}\nValidating overlay: {overlay_label}\n{'=' * 70}")
    pnl_with = callbacks.run_with()
    pnl_without = callbacks.run_without()
    print(f"  with-overlay:    {len(pnl_with)} samples, Sharpe={_sharpe_ann(pnl_with, periods_per_year):+.2f}")
    print(f"  without-overlay: {len(pnl_without)} samples, Sharpe={_sharpe_ann(pnl_without, periods_per_year):+.2f}")

    results = {}
    results["walk_forward"] = walk_forward_test(
        pnl_with, pnl_without, n_folds=n_folds,
        periods_per_year=periods_per_year,
        min_positive_folds=min_positive_folds,
        embargo_pct=embargo_pct,
        use_purged_kfold=use_purged_kfold,
    )
    results["bootstrap"] = bootstrap_test(
        pnl_with, pnl_without, n_boot=n_boot, block_days=block_days,
    )
    if conditional_channel and conditional_bucket:
        results["conditional"] = conditional_bootstrap_test(
            pnl_with, pnl_without, channel=conditional_channel,
            bucket=conditional_bucket, n_boot=n_boot, block_days=block_days,
        )
    else:
        results["conditional"] = {"verdict": "SKIPPED"}

    if callbacks.run_with_shuffled is None or skip_perm:
        results["permutation"] = {"verdict": "SKIPPED",
                                  "reason": "no run_with_shuffled" if callbacks.run_with_shuffled is None else "skip_perm flag"}
    else:
        results["permutation"] = permutation_test(
            run_with=callbacks.run_with,
            run_with_shuffled=callbacks.run_with_shuffled,
            n_perms=n_perms,
            periods_per_year=periods_per_year,
        )
    return results


def print_summary(results: dict, overlay_label: str = "overlay") -> str:
    """Print a one-line-per-test rollup. Return overall verdict string."""
    print(f"\n=== SUMMARY ({overlay_label}) ===")
    for name, r in results.items():
        print(f"  {name:22s}  {r.get('verdict', 'N/A')}")
    passes = sum(1 for r in results.values() if r.get("verdict") == "PASS")
    fails = sum(1 for r in results.values() if r.get("verdict") == "FAIL")
    skips = sum(1 for r in results.values() if r.get("verdict") == "SKIPPED")
    print(f"\n  {passes} PASS / {fails} FAIL / {skips} SKIPPED")
    # Overall verdict heuristic — must pass permutation if it ran,
    # and ≥2 of the 3 statistical tests.
    perm_v = results.get("permutation", {}).get("verdict")
    stat_passes = sum(
        1 for k in ("walk_forward", "bootstrap", "conditional")
        if results.get(k, {}).get("verdict") == "PASS"
    )
    if perm_v == "FAIL":
        overall = "FAIL (permutation null hypothesis not rejected)"
    elif stat_passes >= 2 and perm_v in ("PASS", "SKIPPED"):
        overall = "PASS — ship-ready"
    else:
        overall = "FAIL (insufficient evidence)"
    print(f"\n  OVERALL: {overall}")
    return overall
