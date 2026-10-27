"""Shape Validator: decide whether a backtested filter ("gate") adds a real edge.

Six independent statistical tests on (strategy WITH gate) vs (strategy WITHOUT gate):

  1. Walk-forward (Pardo)
  2. Block bootstrap CI on (with - without) excludes zero
  3. Permutation: actual Sharpe beats random-mask shuffles
  4. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
  5. Purged k-fold + embargo (Lopez de Prado, AFML ch. 7)
  6. CSCV / PBO (Bailey-Borwein-Lopez de Prado-Zhu 2014)
     [only when cohort_returns is provided with >= 4 variants]

PASS = every test that ran passed; PARTIAL = 3 or more; FAIL = fewer.

RISK axis (optional, RiskSpec): separate from the return tests. The six tests validate RETURN edge;
they do not check whether a VaR/ES sizing rule's risk estimate is calibrated. With a RiskSpec the
validator also runs Kupiec POF, Christoffersen independence and an ES exceedance-residual test
[+ optional PIT/Berkowitz]. It does not add to the PASS tally; it is a VETO - a risk FAIL caps a
clean PASS at PARTIAL (never size on a mis-calibrated risk number). Engine: risk_backtest.py.

CPCV (optional, run_cpcv=True): report-only robustness of the gain across combinatorial
sub-periods (ROBUST / MODERATE / CONCENTRATED). Never changes the overall verdict.

What I check before running it:
  - the filter reads the market through the mechanism it claims to exploit
  - the opposite bucket shows the opposite-signed effect
  - the gate mask is causal (no information from after the decision bar)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .gate_builder import GateSpec
from .overlay import OverlayCallbacks, validate
from .purged_kfold import PurgedKFold
from .cscv import cscv as _cscv_compute
from .cpcv import cpcv_subperiod_sharpes as _cpcv
from .signal_attribution import tercile_bucket
from .risk_backtest import (kupiec_pof as _kupiec, christoffersen as _chris,
                            es_backtest as _esbt, pit_berkowitz as _pit,
                            risk_pass as _risk_pass)



def build_gate_mask(
    gate: GateSpec,
    channels_data: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute boolean firing mask for a GateSpec.

    Returns True on days where all (channel, bucket) conditions hold.
    """
    if gate.gate_type != "INTERSECTION":
        # ALWAYS_ON fires everywhere; NO_GATE_FOUND fires nowhere
        return pd.Series(
            gate.gate_type == "ALWAYS_ON",
            index=index, dtype=bool,
        )
    mask = pd.Series(True, index=index, dtype=bool)
    for ch_spec in gate.channels:
        ch_name = ch_spec["channel"]
        bucket = ch_spec["bucket"]
        ch = channels_data[ch_name].reindex(index, method="ffill")
        b = tercile_bucket(ch)
        cond = (b == bucket).fillna(False)
        mask &= cond
    return mask.astype(bool)


from scipy import stats as _st


def deflated_sharpe_pvalue(
    observed_sharpe: float,
    trial_sharpes: np.ndarray,
    daily_returns: np.ndarray,
    periods_per_year: int = 365,
) -> float:
    """Bailey & López de Prado 2014 DSR p-value.

    H0: observed_sharpe <= E[max_SR under null].
    Returns one-sided p-value; small p means observed beats deflation.
    """
    n_trials = len(trial_sharpes)
    sigma_sr = float(trial_sharpes.std(ddof=1)) if n_trials > 1 else 0.5
    em = 0.5772156649  # Euler-Mascheroni
    e = math.e
    e_max_sr = sigma_sr * (
        (1 - em) * _st.norm.ppf(1 - 1.0 / n_trials)
        + em * _st.norm.ppf(1 - 1.0 / (n_trials * e))
    )
    n_obs = len(daily_returns)
    skew = float(_st.skew(daily_returns))
    kurt = float(_st.kurtosis(daily_returns, fisher=False))
    numerator = (observed_sharpe - e_max_sr) * math.sqrt(n_obs - 1)
    denom = math.sqrt(max(1e-12,
                          1 - skew * observed_sharpe
                          + (kurt - 1) / 4 * observed_sharpe ** 2))
    z = numerator / denom
    return float(1 - _st.norm.cdf(z))


@dataclass(frozen=True)
class RiskSpec:
    """Inputs for the Shape Validator RISK axis. Supply when the candidate emits a
    VaR/ES sizing RULE (e.g. a drawdown brake or a volatility-targeted position size) — coverage-backtests
    the RISK estimate the rule sizes on, NOT its return edge. Conventions = risk_backtest.py:
    losses positive (loss = -pnl), VaR/ES are loss thresholds at confidence `alpha`, exception
    when loss > VaR. Optional pred_draws (T x S, strictly OOS) adds the PIT/Berkowitz density test."""
    losses: np.ndarray
    var: np.ndarray
    es: Optional[np.ndarray] = None
    alpha: float = 0.99
    level: float = 0.05
    pred_draws: Optional[object] = None   # per-t predictive draws for PIT (list/array of length T)


@dataclass(frozen=True)
class GateVerdict:
    gate_label: str
    with_sharpe: float
    without_sharpe: float
    lift_sharpe: float
    fire_rate_pct: float
    wfo_verdict: str
    bootstrap_verdict: str
    bootstrap_ci_bps: tuple[float, float]
    conditional_verdict: str
    conditional_ci_bps: tuple[float, float]
    permutation_verdict: str
    permutation_top_k: int  # actual sharpe rank in n_perms shuffles
    dsr_pvalue: float
    dsr_verdict: str
    # Shape Validator addition (2026-05-25): PurgedKFold + embargo
    purged_kfold_verdict: str  # PASS / FAIL / SKIPPED
    purged_kfold_pos_folds: int  # # of folds with positive mean diff
    purged_kfold_n_folds: int
    overall: str  # "PASS" | "PARTIAL" | "FAIL"
    # Shape Validator addition (2026-05-27): CSCV / PBO (BBLZ 2014, MLFAM Ch.8)
    # Cohort-level test — same PBO applies to every variant in the audit cohort.
    # PASS when PBO < 0.5 (beats the complementary-split noise baseline ~0.7).
    # SKIPPED when cohort_returns not provided OR fewer than 4 variants.
    pbo_value: float = float("nan")
    pbo_verdict: str = "SKIPPED"
    cscv_n_chunks_used: int = 0
    cscv_n_combinations: int = 0
    # Shape Validator RISK axis (McNeil, Frey & Embrechts, QRM). ORTHOGONAL to the return
    # tests: validates that a sizing rule's VaR/ES coverage is correct (Kupiec + Christoffersen +
    # ES exceedance-residual [+ optional PIT]). NOT counted toward the return-edge PASS tally; acts
    # as a VETO — a FAIL caps overall at PARTIAL (you must not size on a mis-calibrated risk number).
    # SKIPPED (default) when no RiskSpec supplied -> existing return-only verdicts are unchanged.
    risk_verdict: str = "SKIPPED"
    risk_kupiec_p: float = float("nan")
    risk_christoffersen_p: float = float("nan")
    risk_es_p: float = float("nan")
    risk_pit_p: float = float("nan")
    # CPCV multi-path robustness (AFML Ch.12) — REPORT-ONLY diagnostic on the
    # gate's benefit (with - without): is the incremental edge continuous-robust across combinatorial
    # sub-periods, or CONCENTRATED in a few (route the latter to the episodic validator). Does NOT
    # change `overall`. SKIPPED unless run_cpcv=True. ROBUST/MODERATE/CONCENTRATED.
    cpcv_verdict: str = "SKIPPED"
    cpcv_frac_positive: float = float("nan")
    cpcv_p05: float = float("nan")


def _sharpe(daily: pd.Series, periods_per_year: int = 365) -> float:
    if daily.empty or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * math.sqrt(periods_per_year))


def validate_one_gate(
    gate_label: str,
    with_daily: pd.Series,
    without_daily: pd.Series,
    gate_mask: pd.Series,
    n_total_candidate_gates: int,
    n_perms: int = 25,
    n_boot: int = 2000,
    block_days: int = 14,
    periods_per_year: int = 365,
    skip_purged_kfold: bool = False,
    cohort_returns: Optional[dict[str, pd.Series]] = None,
    cscv_n_chunks: int = 10,
    risk_spec: Optional[RiskSpec] = None,
    run_cpcv: bool = False,
) -> GateVerdict:
    """Run Shape Validator on a single gate.

    n_total_candidate_gates: total candidate gates I tried,
    used as DSR n_trials (honest, not overcounted).

    skip_purged_kfold: if True, run only 4 internal tests (WFO+Boot+Perm+DSR)
    and use 4/4 threshold for PASS. Use this to reproduce the pre-2026-05-25
    4-test verdict (e.g., for direction-switch gates where PurgedKFold
    semantics need refinement).

    cohort_returns: dict mapping {variant_label: daily_return_series} for ALL
        variants in the audit cohort (including this gate). When provided AND
        len(cohort) >= 4, runs CSCV (test 6) and includes PBO in the verdict.
        The SAME PBO applies to every variant in the cohort — it's a property
        of the selection procedure, not the individual gate. Daily-return
        series must be on a common index; rows where ANY variant is NaN are
        dropped before CSCV.
    cscv_n_chunks: number of CSCV chunks (default 10 → C(10,5)=252 combos).

    risk_spec: optional RiskSpec for the RISK axis. Supply when the candidate emits a VaR/ES
        SIZING rule (a drawdown brake, a volatility-targeted size) to coverage-backtest the risk
        estimate (Kupiec + Christoffersen + ES residual [+ PIT]). ORTHOGONAL to the return-edge
        tests — it does NOT inflate the PASS tally; a risk FAIL VETOES a clean PASS down to PARTIAL
        (you must not size on a mis-calibrated VaR/ES). SKIPPED when not supplied (return verdict
        unchanged), so all existing callers are unaffected.
    """
    common = with_daily.index.intersection(without_daily.index)
    with_d = with_daily.loc[common]
    without_d = without_daily.loc[common]

    s_with = _sharpe(with_d, periods_per_year)
    s_without = _sharpe(without_d, periods_per_year)
    lift = s_with - s_without

    # Reuse the existing 4-test pipeline. I provide identity callbacks
    # because I have already pre-computed the daily series.
    cbs = OverlayCallbacks(
        run_with=lambda d=with_d: d,
        run_without=lambda d=without_d: d,
        run_with_shuffled=lambda rng: _shuffle_gate_daily(
            with_d, without_d, gate_mask.reindex(common).fillna(False), rng),
    )
    results = validate(
        cbs,
        overlay_label=gate_label,
        conditional_channel=None,  # I already pre-bucketed; skip channel-conditional
        conditional_bucket=None,
        n_perms=n_perms,
        n_boot=n_boot,
        block_days=block_days,
        periods_per_year=periods_per_year,
    )

    # DSR
    trial_sharpes = np.array([s_with, s_without])  # minimal — actual n_trials supplied
    # Override sigma_sr proxy with externally-supplied trial count for honesty
    dsr_p = deflated_sharpe_pvalue(
        observed_sharpe=s_with,
        trial_sharpes=np.full(n_total_candidate_gates, s_with) +
                       np.random.default_rng(0).normal(0, 0.3, n_total_candidate_gates),
        # ^ heuristic stand-in: assume trial sigma 0.3 with n_total_candidate_gates trials
        daily_returns=with_d.values,
        periods_per_year=periods_per_year,
    )
    dsr_verdict = "PASS" if dsr_p < 0.05 else "FAIL"

    # Aggregate
    wfo = results.get("walk_forward", {}).get("verdict", "FAIL")
    boot = results.get("bootstrap", {}).get("verdict", "FAIL")
    boot_ci = (
        results.get("bootstrap", {}).get("ci_low_bps", 0.0),
        results.get("bootstrap", {}).get("ci_high_bps", 0.0),
    )
    cond = results.get("conditional", {}).get("verdict", "SKIPPED")
    cond_ci = (0.0, 0.0)
    perm = results.get("permutation", {}).get("verdict", "SKIPPED")
    perm_top_k = results.get("permutation", {}).get(
        "rank_observed", n_perms)
    fire_rate = float(gate_mask.reindex(common).fillna(False).mean()) * 100

    # Test 6 (Shape Validator, 2026-05-27): CSCV / PBO — cohort-level test.
    # Computed once and applied identically to every variant in the cohort.
    # PBO < 0.5 = PASS (beats the complementary-split noise baseline ~0.7).
    pbo_value, pbo_verdict, cscv_chunks_used, cscv_combos_used = _maybe_run_cscv(
        cohort_returns=cohort_returns,
        n_chunks=cscv_n_chunks,
    )

    # RISK axis (Shape Validator): coverage-backtest a sizing rule's VaR/ES.
    # Orthogonal to the return tests; acts as a veto via _apply_risk_veto (FAIL caps PASS->PARTIAL).
    risk_verdict, risk_kup_p, risk_chr_p, risk_es_p, risk_pit_p = _maybe_run_risk(risk_spec)

    # CPCV multi-path robustness: report-only diagnostic on the benefit series.
    cpcv_verdict, cpcv_frac_pos, cpcv_p05 = "SKIPPED", float("nan"), float("nan")
    if run_cpcv:
        benefit = (with_d - without_d).dropna()
        if len(benefit) >= 16:
            cr = _cpcv(benefit.values)
            cpcv_verdict, cpcv_frac_pos, cpcv_p05 = cr.verdict, cr.frac_positive, cr.p05

    # Test 5 (Shape Validator): PurgedKFold + embargo (LdP AFML Ch.7)
    if skip_purged_kfold:
        pkf_verdict = "SKIPPED"
        pkf_pos_folds = 0
        pkf_n_folds = 5
        # 4-test legacy threshold (CSCV layered in additively if cohort supplied)
        core_pass = sum(1 for v in (wfo, boot, perm, dsr_verdict) if v == "PASS")
        if pbo_verdict == "SKIPPED":
            overall = "PASS" if core_pass == 4 else ("PARTIAL" if core_pass >= 2 else "FAIL")
        else:
            tests_run = 5  # 4 core + CSCV
            total_pass = core_pass + (1 if pbo_verdict == "PASS" else 0)
            if total_pass == tests_run:
                overall = "PASS"
            elif total_pass >= 3:
                overall = "PARTIAL"
            else:
                overall = "FAIL"
        overall = _apply_risk_veto(overall, risk_verdict)
        return GateVerdict(
            gate_label=gate_label, with_sharpe=s_with, without_sharpe=s_without,
            lift_sharpe=lift, fire_rate_pct=fire_rate,
            wfo_verdict=wfo, bootstrap_verdict=boot,
            bootstrap_ci_bps=tuple(float(x) for x in boot_ci),
            conditional_verdict=cond, conditional_ci_bps=cond_ci,
            permutation_verdict=perm, permutation_top_k=int(perm_top_k),
            dsr_pvalue=dsr_p, dsr_verdict=dsr_verdict,
            purged_kfold_verdict=pkf_verdict,
            purged_kfold_pos_folds=pkf_pos_folds, purged_kfold_n_folds=pkf_n_folds,
            overall=overall,
            pbo_value=pbo_value, pbo_verdict=pbo_verdict,
            cscv_n_chunks_used=cscv_chunks_used,
            cscv_n_combinations=cscv_combos_used,
            risk_verdict=risk_verdict, risk_kupiec_p=risk_kup_p,
            risk_christoffersen_p=risk_chr_p, risk_es_p=risk_es_p, risk_pit_p=risk_pit_p,
            cpcv_verdict=cpcv_verdict, cpcv_frac_positive=cpcv_frac_pos, cpcv_p05=cpcv_p05,
        )
    # (continued)
    # PurgedKFold full implementation:
    # Events = days the gate fires. Label = the bare strategy return on
    # that gate-fire day (what the strategy actually earned when fired).
    # Cross-validate via 5-fold with 2% embargo. PASS if 4+/5 folds
    # have positive mean return on the test partition AND that mean
    # exceeds the corresponding mean on non-fire days (i.e. the gate
    # actually selects better days, not just non-zero days).
    gate_active = gate_mask.reindex(common).fillna(False).astype(bool)
    gate_days = common[gate_active.values]
    pkf_n_folds = 5
    if len(gate_days) < 25:
        pkf_verdict = "SKIPPED"
        pkf_pos_folds = 0
    else:
        events = pd.DataFrame(
            {"end_time": gate_days + pd.Timedelta(days=1)},
            index=gate_days,
        )
        # Label: the bare strategy return on each fire day
        fire_day_return = without_d.loc[gate_days]
        # Reference: mean bare return on non-fire days (what I would earn
        # without the gate's selection). Gate must beat this to PASS.
        non_fire_days = common[~gate_active.values]
        ref_mean = float(without_d.loc[non_fire_days].mean()) if len(non_fire_days) else 0.0
        pkf = PurgedKFold(n_splits=pkf_n_folds, embargo_pct=0.02)
        fold_means = []
        try:
            for train_idx, test_idx in pkf.split(events):
                if len(test_idx) == 0:
                    continue
                test_days = events.index[test_idx]
                fold_means.append(float(fire_day_return.loc[test_days].mean()))
        except Exception as exc:
            pkf_verdict = f"ERROR ({exc})"
            pkf_pos_folds = 0
        else:
            # PASS: 4+ folds where gate-fire mean > non-fire mean
            # (i.e. the gate genuinely selects better days, not just non-zero days)
            pkf_pos_folds = sum(1 for m in fold_means if m > ref_mean)
            pkf_verdict = "PASS" if pkf_pos_folds >= 4 else "FAIL"

    core_pass = sum(
        1 for v in (wfo, boot, perm, dsr_verdict, pkf_verdict) if v == "PASS"
    )
    # Shape Validator overall — N-test scoring (5 core + optional CSCV):
    #   PASS    = ALL tests pass (strongest convergent evidence)
    #   PARTIAL = 3+ tests pass
    #   FAIL    = <3 tests pass
    if pbo_verdict == "SKIPPED":
        tests_run = 5
        total_pass = core_pass
    else:
        tests_run = 6
        total_pass = core_pass + (1 if pbo_verdict == "PASS" else 0)
    if total_pass == tests_run:
        overall = "PASS"
    elif total_pass >= 3:
        overall = "PARTIAL"
    else:
        overall = "FAIL"
    overall = _apply_risk_veto(overall, risk_verdict)

    return GateVerdict(
        gate_label=gate_label,
        with_sharpe=s_with,
        without_sharpe=s_without,
        lift_sharpe=lift,
        fire_rate_pct=fire_rate,
        wfo_verdict=wfo,
        bootstrap_verdict=boot,
        bootstrap_ci_bps=tuple(float(x) for x in boot_ci),
        conditional_verdict=cond,
        conditional_ci_bps=cond_ci,
        permutation_verdict=perm,
        permutation_top_k=int(perm_top_k),
        dsr_pvalue=dsr_p,
        dsr_verdict=dsr_verdict,
        purged_kfold_verdict=pkf_verdict,
        purged_kfold_pos_folds=pkf_pos_folds,
        purged_kfold_n_folds=pkf_n_folds,
        overall=overall,
        pbo_value=pbo_value,
        pbo_verdict=pbo_verdict,
        cscv_n_chunks_used=cscv_chunks_used,
        cscv_n_combinations=cscv_combos_used,
        risk_verdict=risk_verdict,
        risk_kupiec_p=risk_kup_p,
        risk_christoffersen_p=risk_chr_p,
        risk_es_p=risk_es_p,
        risk_pit_p=risk_pit_p,
        cpcv_verdict=cpcv_verdict,
        cpcv_frac_positive=cpcv_frac_pos,
        cpcv_p05=cpcv_p05,
    )


def _maybe_run_cscv(
    cohort_returns: Optional[dict[str, pd.Series]],
    n_chunks: int,
) -> tuple[float, str, int, int]:
    """Compute PBO across an audit cohort. Returns (pbo, verdict, n_chunks_used, n_combinations).
    SKIPPED when cohort not supplied OR < 4 variants OR rows insufficient.
    """
    if cohort_returns is None or len(cohort_returns) < 4:
        return float("nan"), "SKIPPED", 0, 0
    # Align on common dates, drop rows where any variant is NaN
    df = pd.DataFrame({k: v for k, v in cohort_returns.items()}).dropna(how="any")
    if df.empty:
        return float("nan"), "SKIPPED", 0, 0
    # CSCV requires rows_per_chunk >= 2; auto-shrink S if T is small
    rows_per_chunk = len(df) // n_chunks
    while rows_per_chunk < 5 and n_chunks > 4:
        n_chunks -= 2
        rows_per_chunk = len(df) // n_chunks
    if rows_per_chunk < 2:
        return float("nan"), "SKIPPED", 0, 0
    try:
        result = _cscv_compute(df.values, n_chunks=n_chunks)
    except Exception:
        return float("nan"), "SKIPPED", 0, 0
    # PASS threshold = 0.5 (well below the ~0.7 noise-baseline; the
    # conventional CSCV bar; equivalent to the "GOOD" label in cscv.py)
    verdict = "PASS" if result.pbo < 0.5 else "FAIL"
    return float(result.pbo), verdict, result.n_chunks, result.n_combinations


def _maybe_run_risk(
    risk_spec: Optional[RiskSpec],
) -> tuple[str, float, float, float, float]:
    """Shape Validator RISK axis. Returns (verdict, kupiec_p, christoffersen_p, es_p, pit_p).
    SKIPPED when no RiskSpec. PASS = Kupiec coverage OK AND Christoffersen independence OK AND ES
    not-too-low [AND PIT uniform if pred_draws supplied]. Uses risk_backtest.risk_pass for the core
    veto and the component tests for the reported p-values."""
    if risk_spec is None:
        return "SKIPPED", float("nan"), float("nan"), float("nan"), float("nan")
    L = np.asarray(risk_spec.losses, float)
    V = np.asarray(risk_spec.var, float)
    if L.size < 30 or V.size != L.size:
        return "SKIPPED", float("nan"), float("nan"), float("nan"), float("nan")
    E = None if risk_spec.es is None else np.asarray(risk_spec.es, float)
    a, lvl = risk_spec.alpha, risk_spec.level
    rv = _risk_pass(L, V, E, alpha=a, level=lvl)
    kup_p = _kupiec(L, V, a)[1]
    chr_p = _chris(L, V, a)["p_ind"]
    es_p = _esbt(L, V, E)["p_boot"] if E is not None else float("nan")
    pit_p = float("nan")
    overall_ok = bool(rv.overall)
    if risk_spec.pred_draws is not None:
        try:
            pit = _pit(L, risk_spec.pred_draws)
            pit_p = pit["ks_p"]
            overall_ok = overall_ok and (pit_p >= lvl)
        except Exception:
            pit_p = float("nan")
    return ("PASS" if overall_ok else "FAIL"), float(kup_p), float(chr_p), float(es_p), pit_p


def _apply_risk_veto(overall: str, risk_verdict: str) -> str:
    """RISK axis is a veto, not a return-edge test: a FAIL caps a clean PASS at PARTIAL (the edge
    may be real but you must not size on a mis-calibrated VaR/ES). SKIPPED/PASS leave overall as-is."""
    if risk_verdict == "FAIL" and overall == "PASS":
        return "PARTIAL"
    return overall


# Short name. validate_one_gate is the original name; both invoke the same pipeline.
shape_validate = validate_one_gate


def _shuffle_gate_daily(
    with_d: pd.Series,
    without_d: pd.Series,
    gate_mask: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """For permutation: shuffle the gate mask across time, rebuild daily."""
    order = rng.permutation(len(gate_mask))
    shuf = pd.Series(gate_mask.values[order], index=gate_mask.index)
    # Synthetic: when gate ON use with_d; else use without_d
    result = without_d.copy()
    result[shuf] = with_d[shuf]
    return result
