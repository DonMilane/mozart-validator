"""Does the validator separate noise from an edge, and where is the line?

How the synthetic data is built matters more than the numbers in it. The gated strategy gets its OWN
daily noise on gate-on days, and each cohort variant gets its own as well. If "with" were just
"without" plus a planted constant, the difference between them would contain no noise at all and any
edge, however small, would look perfectly significant. That is how an earlier version of this file
concluded the validator could not tell 1 bp/day from 25 bp/day. With honest noise it can.

Calibration (8 seeds each, 950 days, 1.2% daily vol, gate on 45% of days, 5 candidate gates):

    edge/day   ~a year    FAIL  PARTIAL  PASS
      0 bp       0%         6      2      0
      1 bp      1.6%        5      3      0
      5 bp      8.2%        4      4      0
     10 bp     16.4%        2      6      0
     15 bp     24.6%        0      8      0
     25 bp     41.1%        0      8      0

What the tests pin down:

  test_noise_never_passes     no edge: never PASS, mostly FAIL. Noise does reach PARTIAL on 2 of 8
                              seeds, so PARTIAL alone is weak evidence.
  test_clear_edge_detected    a 15 bp/day edge (about 25% a year, t ~ 2.6) never FAILs.
  test_verdict_tracks_size    a 1 bp/day edge (t ~ 0.2, invisible in this much noise) mostly FAILs
                              while the 15 bp edge does not.

Nothing reaches PASS here: four or more tests passing on 950 noisy days with 5 candidates is a high
bar, which is the point of PASS.

Run: pytest -q tests/
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from shape_validator import shape_validate

SEEDS = range(8)


def _verdict(effect_bps: float, seed: int, n: int = 950) -> str:
    """One synthetic gate. On gate-on days each variant earns `effect_bps` (scaled per variant) plus
    its own noise; the variant at scale 1.0 is the gate under test."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    mask = pd.Series(rng.random(n) < 0.45, index=idx)
    on = mask.astype(float)
    without = pd.Series(rng.normal(0, 0.012, n), index=idx)
    cohort = {f"v{i}": without + on * (effect_bps * (0.6 + 0.2 * i) / 1e4)
              + pd.Series(rng.normal(0, 0.012, n), index=idx) * on
              for i in range(5)}
    with contextlib.redirect_stdout(io.StringIO()):          # the validator is chatty
        v = shape_validate(gate_label=f"synthetic_{effect_bps}bp",
                           with_daily=cohort["v2"], without_daily=without, gate_mask=mask,
                           n_total_candidate_gates=5, cohort_returns=cohort, n_perms=25)
    return v.overall


def test_noise_never_passes():
    """No edge at all. A PASS here would mean the validator certifies noise - the one failure that
    makes it worse than useless."""
    verdicts = [_verdict(0.0, s) for s in SEEDS]
    assert "PASS" not in verdicts, f"noise PASSED: {verdicts}"
    assert verdicts.count("FAIL") >= 5, f"noise not mostly rejected: {verdicts}"


def test_clear_edge_detected():
    """15 bp/day on 45% of days, about 25% a year: clearly there, must never be thrown away."""
    verdicts = [_verdict(15.0, s) for s in SEEDS]
    assert "FAIL" not in verdicts, f"15 bp edge missed: {verdicts}"


def test_verdict_tracks_size():
    """A 1 bp/day edge is statistically invisible in 1.2% daily vol over ~430 gate-on days
    (t ~ 0.2); it should mostly FAIL, unlike the 15 bp edge."""
    small = [_verdict(1.0, s) for s in SEEDS]
    assert small.count("FAIL") >= 4, f"1 bp edge graded like a real one: {small}"
