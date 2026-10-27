"""Quickstart: grade one filter on a synthetic strategy, then print the verdict card.

Run:  python examples/quickstart.py
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from shape_validator import shape_validate

rng = np.random.default_rng(4)
idx = pd.date_range("2022-01-01", periods=950, freq="D")

# The strategy without the filter: noisy daily returns.
without = pd.Series(rng.normal(0.0006, 0.012, len(idx)), index=idx)

# The filter: on about 45% of days. On those days the strategy earns 5bp more.
mask = pd.Series(rng.random(len(idx)) < 0.45, index=idx)
with_ = without + mask.astype(float) * 0.0005

# Every variant I tried (here: five strengths of the same filter). Enables test 6.
cohort = {f"v{i}": without + mask.astype(float) * 0.0005 * (0.6 + 0.2 * i) for i in range(5)}

with contextlib.redirect_stdout(io.StringIO()):   # the tests print their working; hide it
    v = shape_validate(
        gate_label="my_filter",
        with_daily=with_,
        without_daily=without,
        gate_mask=mask,
        n_total_candidate_gates=5,   # how many variants I tried - honestly
        cohort_returns=cohort,
        periods_per_year=365,        # 365 for crypto, 252 for stocks
    )

print(f"overall          {v.overall}")
print(f"Sharpe           {v.without_sharpe:.2f} -> {v.with_sharpe:.2f} (lift {v.lift_sharpe:+.2f})")
print(f"filter on        {v.fire_rate_pct:.0f}% of days")
print(f"1 walk-forward   {v.wfo_verdict}")
print(f"2 bootstrap      {v.bootstrap_verdict}  CI {v.bootstrap_ci_bps[0]:+.1f} .. {v.bootstrap_ci_bps[1]:+.1f} bp")
print(f"3 permutation    {v.permutation_verdict}  rank {v.permutation_top_k}")
print(f"4 deflated SR    {v.dsr_verdict}  p={v.dsr_pvalue:.3f}")
print(f"5 purged k-fold  {v.purged_kfold_verdict}  {v.purged_kfold_pos_folds}/{v.purged_kfold_n_folds} folds")
print(f"6 CSCV / PBO     {v.pbo_verdict}  PBO={v.pbo_value:.2f}")
