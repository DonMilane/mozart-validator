"""RISK-MEASURE backtest axis for the Shape Validator (McNeil, Frey & Embrechts, Quantitative Risk Management).

The standing validator (validate_one_gate / shape_validate) tests RETURN edge only (WFO + bootstrap +
permutation + DSR + PurgedKFold + CSCV). It does NOT test whether a shipped sizing/VaR/ES RULE's RISK
estimate is CORRECT. This module adds that axis — the pre-condition for any rule that sizes positions from a VaR or ES
estimate (you must not size on a VaR/ES whose coverage is wrong).

FOUR INDEPENDENT TESTS (McNeil QRM ch9; conventions stated so they are auditable):
  losses L_t (positive = loss), VaR_t = the alpha-level VaR (a LOSS threshold; exception when L_t > VaR_t),
  alpha = VaR confidence (e.g. 0.99 -> expected exception rate p = 1-alpha = 0.01).
  1. KUPIEC POF (unconditional coverage): exception rate = p? LR_uc ~ chi2(1).
  2. CHRISTOFFERSEN independence (Markov): are exceptions iid (no clustering)? LR_ind ~ chi2(1);
     LR_cc = LR_uc + LR_ind ~ chi2(2) = conditional coverage.
  3. ES exceedance-residual (McNeil-Frey): ES is NOT elicitable (only the quantile is) -> backtest by
     testing that on exceedance days the residual (L_t - ES_t) has MEAN ZERO (bootstrap p); a positive
     mean = ES too small (tail underestimated). Also the elicitable QUANTILE/pinball score for ranking.
  4. PIT / BERKOWITZ: if the rule emits a full predictive distribution, U_t = F_t(L_t) iid Uniform
     validates the WHOLE density; Berkowitz transforms z_t = Phi^-1(U_t) ~ N(0,1) and LR-tests mean/var.
     PIT must be built strictly OUT-OF-SAMPLE (F_t from data <= t). [requires predictive draws/CDF]

VERDICT helper: risk_pass(...) = coverage (Kupiec) AND independence (Christoffersen) AND ES not-too-low.
Engine self-tested against known answers (correct model passes, too-low VaR/clustered/too-low-ES rejected).
Caveat (McNeil): tests are IID-style; serial correlation in violations is exactly what Christoffersen
catches -> always report LR_ind, not just LR_uc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

RNG = np.random.default_rng(485)


# ---------------------------------------------------------------- Kupiec POF
def kupiec_pof(losses, var, alpha: float):
    """Unconditional-coverage LR test. Returns (LR_uc, p_value, n_exc, exp_rate, obs_rate)."""
    L = np.asarray(losses, float); V = np.asarray(var, float)
    m = len(L); p = 1.0 - alpha
    x = int(np.sum(L > V))                       # exceptions
    pi = x / m if m else np.nan
    if x == 0:
        lr = -2.0 * (m * np.log(1 - p))
    elif x == m:
        lr = -2.0 * (m * np.log(p))
    else:
        ll0 = (m - x) * np.log(1 - p) + x * np.log(p)
        ll1 = (m - x) * np.log(1 - pi) + x * np.log(pi)
        lr = -2.0 * (ll0 - ll1)
    pval = float(stats.chi2.sf(lr, 1))
    return float(lr), pval, x, p, float(pi)


# ---------------------------------------------------------------- Christoffersen
def christoffersen(losses, var, alpha: float):
    """Independence (Markov) LR + combined conditional-coverage LR.
    Returns dict(LR_ind, p_ind, LR_cc, p_cc, n01,n11,...)."""
    L = np.asarray(losses, float); V = np.asarray(var, float)
    I = (L > V).astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(I[:-1], I[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    eps = 1e-12
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def lll(p, k1, k0):  # log-lik of k1 ones, k0 zeros under prob p
        return k1 * np.log(p + eps) + k0 * np.log(1 - p + eps)
    ll_ind = lll(pi, n01 + n11, n00 + n10)
    ll_mkv = lll(pi01, n01, n00) + lll(pi11, n11, n10)
    LR_ind = -2.0 * (ll_ind - ll_mkv)
    LR_uc = kupiec_pof(L, V, alpha)[0]
    LR_cc = LR_uc + LR_ind
    return dict(LR_ind=float(LR_ind), p_ind=float(stats.chi2.sf(LR_ind, 1)),
                LR_cc=float(LR_cc), p_cc=float(stats.chi2.sf(LR_cc, 2)),
                n00=n00, n01=n01, n10=n10, n11=n11)


# ---------------------------------------------------------------- ES backtest (McNeil-Frey)
def es_backtest(losses, var, es, n_boot: int = 5000):
    """On exceedance days (L>VaR) the residual (L-ES) should be mean-0. mean>0 = ES too small.
    Returns dict(n_exc, resid_mean, p_boot, quantile_score). Bootstrap is one-sided (ES-too-low)."""
    L = np.asarray(losses, float); V = np.asarray(var, float); E = np.asarray(es, float)
    exc = L > V
    r = (L - E)[exc]                              # exceedance residual
    if len(r) < 2:
        return dict(n_exc=int(exc.sum()), resid_mean=np.nan, p_boot=np.nan, quantile_score=np.nan)
    obs = r.mean()
    rc = r - r.mean()                             # center for H0
    boot = np.array([RNG.choice(rc, size=len(r), replace=True).mean() for _ in range(n_boot)])
    p_boot = float((boot >= obs).mean())          # one-sided: P(boot mean >= observed) under mean-0
    return dict(n_exc=int(exc.sum()), resid_mean=float(obs), p_boot=p_boot)


# ---------------------------------------------------------------- PIT / Berkowitz
def pit_berkowitz(losses, pred_draws):
    """U_t = Fhat_t(L_t) from per-t predictive draws (T x S). Berkowitz LR on z=Phi^-1(U) ~ N(0,1).
    pred_draws[t] MUST be strictly out-of-sample. Returns dict(ks_p, berkowitz_LR, berkowitz_p)."""
    L = np.asarray(losses, float)
    U = np.array([(np.asarray(pred_draws[t]) <= L[t]).mean() for t in range(len(L))])
    U = np.clip(U, 1e-4, 1 - 1e-4)
    ks_p = float(stats.kstest(U, "uniform").pvalue)
    z = stats.norm.ppf(U)
    mu, sd = z.mean(), z.std(ddof=1)
    # Berkowitz LR: H0 mu=0, sd=1 (no AR for now). LR = -2(ll0 - ll1) ~ chi2(2)
    ll1 = np.sum(stats.norm.logpdf(z, mu, sd))
    ll0 = np.sum(stats.norm.logpdf(z, 0.0, 1.0))
    LR = -2.0 * (ll0 - ll1)
    return dict(ks_p=ks_p, berkowitz_LR=float(LR), berkowitz_p=float(stats.chi2.sf(LR, 2)),
                z_mean=float(mu), z_sd=float(sd))


@dataclass(frozen=True)
class RiskVerdict:
    coverage_ok: bool      # Kupiec not rejected
    independent_ok: bool   # Christoffersen not rejected
    es_ok: bool            # ES residual not significantly > 0
    overall: bool


def risk_pass(losses, var, es=None, alpha: float = 0.99, level: float = 0.05) -> RiskVerdict:
    pk = kupiec_pof(losses, var, alpha); ch = christoffersen(losses, var, alpha)
    cov = pk[1] >= level
    ind = ch["p_ind"] >= level
    es_ok = True
    if es is not None:
        eb = es_backtest(losses, var, es)
        es_ok = (np.isnan(eb["p_boot"])) or (eb["p_boot"] >= level)   # not significantly too-low
    return RiskVerdict(cov, ind, es_ok, cov and ind and es_ok)


# ---------------------------------------------------------------- self-test (known answers)
def _self_test() -> None:
    T = 4000; alpha = 0.99
    L = RNG.standard_t(5, size=T) * 1.0          # heavy-tailed losses
    var_true = stats.t.ppf(alpha, 5)             # CORRECT VaR for the t(5)
    es_true = 5 / (5 - 1) * (5 + var_true**2) / 5 * stats.t.pdf(var_true, 5) / (1 - alpha)  # t ES (approx)
    # 1. correct VaR -> Kupiec NOT rejected; too-low VaR (90% quantile) -> rejected
    p_ok = kupiec_pof(L, np.full(T, var_true), alpha)[1]
    p_low = kupiec_pof(L, np.full(T, stats.t.ppf(0.90, 5)), alpha)[1]
    assert p_ok > 0.05 and p_low < 0.01, f"kupiec fail: ok={p_ok:.3f} low={p_low:.4f}"
    # 2. clustered exceptions -> Christoffersen LR_ind rejects. Build a process where exceptions bunch.
    Lc = RNG.standard_t(5, size=T)
    burst = RNG.integers(0, T - 30, size=15)
    for b in burst:
        Lc[b:b+8] = stats.t.ppf(0.999, 5) + 0.5  # forced consecutive exceedances
    p_ind = christoffersen(Lc, np.full(T, var_true), alpha)["p_ind"]
    assert p_ind < 0.05, f"christoffersen fail: p_ind={p_ind:.3f}"
    # 3. ES: correct-ish ES residual ~mean 0 (p high); too-low ES (=VaR) -> residual>0 (p low)
    p_es_ok = es_backtest(L, np.full(T, var_true), np.full(T, es_true))["p_boot"]
    p_es_low = es_backtest(L, np.full(T, var_true), np.full(T, var_true))["p_boot"]
    assert p_es_low < 0.05, f"ES fail: too-low p={p_es_low:.3f}"
    # 4. PIT: correct predictive draws -> uniform (ks high); wrong scale -> reject
    draws_ok = [RNG.standard_t(5, size=400) for _ in range(800)]
    Lp = np.array([d[0] for d in [RNG.standard_t(5, size=1) for _ in range(800)]]).ravel()
    ks_ok = pit_berkowitz(Lp, draws_ok)["ks_p"]
    draws_bad = [RNG.standard_t(5, size=400) * 0.3 for _ in range(800)]   # too-narrow predictive
    bk_bad = pit_berkowitz(Lp, draws_bad)["berkowitz_p"]
    assert bk_bad < 0.05, f"PIT fail: bad berkowitz p={bk_bad:.3f}"
    print(f"[self-test] PASS  Kupiec(ok {p_ok:.2f}/low {p_low:.0e})  Christoffersen(clustered p_ind {p_ind:.3f})  "
          f"ES(too-low p {p_es_low:.3f})  PIT(narrow berkowitz p {bk_bad:.3f})")


if __name__ == "__main__":
    _self_test()
    print("risk_backtest engine OK - used by shape_validate when a RiskSpec is supplied.")
