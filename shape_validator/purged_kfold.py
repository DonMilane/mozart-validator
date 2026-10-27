"""PurgedKFold — time-series cross-validation with purge + embargo.

Pure-pandas/numpy implementation. License-free reimplementation of the
methodology from López de Prado "Advances in Financial ML" Ch. 7 and
Gatto 2025 "Within-Strategy Permutation Tests".

Why this exists:
  Naive k-fold cross-validation LEAKS in time-series ML because:
  1. Trade labels are determined over a horizon (e.g., 96h triple-barrier).
     Training events that started before the test fold but ended INSIDE
     the test fold see future information.
  2. Returns are serially correlated. Training events immediately after
     the test fold carry residual information from the test period.

  PurgedKFold fixes both:
  - PURGE: drop training events whose label-window endtime falls inside
           the test fold's time range.
  - EMBARGO: drop training events whose start time falls within an
             embargo period (e.g., 1% of total span) AFTER the test fold ends.

Use as a context manager / generator yielding (train_idx, test_idx) pairs.

Example:
    >>> events = pd.DataFrame({"end_time": close_times}, index=open_times)
    >>> splitter = PurgedKFold(n_splits=5, embargo_pct=0.01)
    >>> for train_idx, test_idx in splitter.split(events):
    ...     train_events = events.iloc[train_idx]
    ...     test_events = events.iloc[test_idx]
"""
from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd


class PurgedKFold:
    """Time-series K-fold with purge + embargo (LdP AFML Ch.7, Gatto 2025).

    Args:
        n_splits: number of folds (default 5)
        embargo_pct: embargo as fraction of total time span (default 0.01 = 1%)
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be ≥ 2")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, events: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) integer-location index pairs.

        Args:
            events: DataFrame indexed by event START time, with a column
                    named `end_time` giving each event's label-window end.

        Yields:
            (train_idx, test_idx) — np.ndarray of integer positions into events.
        """
        if "end_time" not in events.columns:
            raise ValueError("events must have 'end_time' column")
        if not events.index.is_monotonic_increasing:
            raise ValueError("events index must be sorted by start time")

        n = len(events)
        end_times = events["end_time"].values
        start_times = events.index.values

        # Compute embargo span in nanoseconds (works for tz-aware DatetimeIndex)
        span_ns = int(start_times[-1].astype("datetime64[ns]").astype(np.int64) -
                      start_times[0].astype("datetime64[ns]").astype(np.int64))
        embargo_ns = int(self.embargo_pct * span_ns)

        # Split test set into n_splits contiguous blocks by event count
        # (simpler + more balanced than splitting by time when event density varies)
        fold_size = n // self.n_splits
        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end)

            test_t0 = pd.Timestamp(start_times[test_start])
            test_t1 = pd.Timestamp(end_times[test_end - 1])

            # All other indices are CANDIDATES for training
            train_candidates = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n)
            ])

            # PURGE: drop training events whose end_time falls inside [test_t0, test_t1]
            # i.e., events whose label horizon overlaps the test period
            purged = []
            for i in train_candidates:
                ev_start = pd.Timestamp(start_times[i])
                ev_end = pd.Timestamp(end_times[i])
                # Event overlaps test if start ≤ test_t1 AND end ≥ test_t0
                if not (ev_end < test_t0 or ev_start > test_t1):
                    continue   # overlap — purge
                # EMBARGO: drop training events whose start_time is within
                # embargo period AFTER test_t1
                if pd.Timestamp(ev_start) > test_t1 and \
                   (ev_start.value - test_t1.value) < embargo_ns:
                    continue
                purged.append(i)
            yield np.array(purged), test_idx


def purged_kfold_report(events: pd.DataFrame,
                         metric_fn,
                         n_splits: int = 5,
                         embargo_pct: float = 0.01) -> pd.DataFrame:
    """Run PurgedKFold and aggregate a per-fold metric.

    Args:
        events: DataFrame with `end_time` column, indexed by start time.
        metric_fn: callable(train_events, test_events) -> dict of metrics.
        n_splits, embargo_pct: forwarded to PurgedKFold.

    Returns:
        DataFrame with one row per fold + aggregate row.
    """
    splitter = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(events)):
        train_events = events.iloc[train_idx]
        test_events = events.iloc[test_idx]
        m = metric_fn(train_events, test_events)
        m["fold"] = fold
        m["n_train"] = len(train_idx)
        m["n_test"] = len(test_idx)
        rows.append(m)
    df = pd.DataFrame(rows).set_index("fold")
    return df


# Self-test / smoke check
def _selftest():
    """Smoke test: synthetic time-series events, verify purge + embargo behavior."""
    n = 100
    start = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    # Each event's label window is 7 days
    end = start + pd.Timedelta(days=7)
    events = pd.DataFrame({"end_time": end}, index=start)

    splitter = PurgedKFold(n_splits=5, embargo_pct=0.02)
    for fold, (tr, te) in enumerate(splitter.split(events)):
        t0 = events.index[te[0]]
        t1 = events["end_time"].iloc[te[-1]]
        # Verify no train events overlap test window
        tr_starts = events.index[tr]
        tr_ends = events["end_time"].iloc[tr]
        overlaps = ((tr_starts <= t1) & (tr_ends >= t0)).sum()
        assert overlaps == 0, f"Fold {fold}: {overlaps} train events overlap test window"
        # Verify embargo
        embargo_ns = int(0.02 * (events.index[-1].value - events.index[0].value))
        post_test_train = tr_starts[tr_starts > t1]
        if len(post_test_train) > 0:
            gap = (post_test_train[0].value - t1.value)
            assert gap >= embargo_ns, f"Fold {fold}: embargo violated, gap {gap}ns < {embargo_ns}ns"
        print(f"  Fold {fold}: n_train={len(tr)}, n_test={len(te)}, "
              f"test_period={t0.date()} → {t1.date()}, embargo OK")
    print("PurgedKFold self-test PASSED")


if __name__ == "__main__":
    _selftest()
