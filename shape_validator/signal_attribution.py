"""Module A: per-signal × per-channel daily-lift attribution.

For each (signal, channel) pair: bucket days by channel tercile, compute
mean lift + bootstrap CI per bucket.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


def tercile_bucket(s: pd.Series) -> pd.Series:
    """Assign each value a LOW/MID/HIGH label by tercile.
    NaN values stay NaN."""
    valid = s.dropna()
    q33, q67 = valid.quantile([1 / 3, 2 / 3])
    out = pd.Series(index=s.index, dtype=object)
    out[s <= q33] = "LOW"
    out[(s > q33) & (s < q67)] = "MID"
    out[s >= q67] = "HIGH"
    out[s.isna()] = np.nan
    return out


def block_bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = 2000,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Block bootstrap 95% CI on the mean of arr.

    Preserves serial correlation by sampling blocks of contiguous values.
    """
    n = len(arr)
    if n < block_size:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block_size)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_size] for s in starts])
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class AttributionRow:
    signal: str
    channel: str
    bucket: str            # "LOW" | "MID" | "HIGH"
    n_days: int
    mean_bps: float
    ci_lo_bps: float
    ci_hi_bps: float
    significant: bool      # True iff CI excludes zero AND mean > 0


def attribute_signal_channel(
    signal_name: str,
    channel_name: str,
    diff_daily: pd.Series,
    channel_daily: pd.Series,
    n_boot: int = 2000,
    block_size: int = 5,
) -> list[AttributionRow]:
    """Per (signal, channel): 3 rows (LOW/MID/HIGH) with bootstrap-CI lift."""
    # Align indices
    common = diff_daily.index.intersection(channel_daily.index)
    d = diff_daily.loc[common].dropna()
    c = channel_daily.loc[common].reindex(d.index).dropna()
    d = d.loc[c.index]
    buckets = tercile_bucket(c)

    rows: list[AttributionRow] = []
    for bucket in ("LOW", "MID", "HIGH"):
        mask = (buckets == bucket).values
        sub = d.values[mask]
        n = len(sub)
        if n == 0:
            rows.append(AttributionRow(
                signal=signal_name, channel=channel_name, bucket=bucket,
                n_days=0, mean_bps=0.0, ci_lo_bps=0.0, ci_hi_bps=0.0,
                significant=False,
            ))
            continue
        mean_bps = float(sub.mean()) * 1e4
        ci_lo, ci_hi = block_bootstrap_ci(sub, n_boot=n_boot,
                                           block_size=block_size,
                                           seed=hash((signal_name, channel_name, bucket)) % (2**32))
        ci_lo_bps = ci_lo * 1e4
        ci_hi_bps = ci_hi * 1e4
        sig = (ci_lo_bps > 0) and (mean_bps > 0)
        rows.append(AttributionRow(
            signal=signal_name, channel=channel_name, bucket=bucket,
            n_days=n, mean_bps=mean_bps, ci_lo_bps=ci_lo_bps,
            ci_hi_bps=ci_hi_bps, significant=sig,
        ))
    return rows


def build_attribution_table(
    diff_panels: dict[str, pd.Series],
    channel_panels: dict[str, pd.Series],
    n_boot: int = 2000,
    block_size: int = 5,
) -> pd.DataFrame:
    """Iterate all (signal, channel) pairs; emit a flat DataFrame.

    diff_panels: {signal_name: daily_diff_series}
    channel_panels: {channel_name: daily_channel_series}
    """
    all_rows: list[AttributionRow] = []
    for signal_name, diff in diff_panels.items():
        for channel_name, channel in channel_panels.items():
            rows = attribute_signal_channel(
                signal_name, channel_name, diff, channel,
                n_boot=n_boot, block_size=block_size,
            )
            all_rows.extend(rows)
    return pd.DataFrame([r.__dict__ for r in all_rows])
