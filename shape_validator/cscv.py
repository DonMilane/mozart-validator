"""CSCV — Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, López de Prado, Zhu (2014) "The Probability of Backtest
Overfitting", Journal of Computational Finance; reprinted in 'Machine
Learning for Asset Managers' (2020) Ch.8.

Why I need this
----------------
DSR (already in Shape Validator) handles multiple-testing correction
under known n_trials. CSCV answers a complementary question: given a
SELECTION procedure that picks the best-IS strategy from M variants,
what's the probability the chosen one underperforms its peers OOS?
That's the **Probability of Backtest Overfitting (PBO)** — the rate at
which my selection rule prefers strategies that disappear OOS.

Important: the null-baseline (pure noise) is NOT PBO=0.5. CSCV uses
COMPLEMENTARY IS/OOS splits (not random holdout), so by construction
IS-mean + OOS-mean = 2·overall-mean per strategy. The IS-best strategy
(positive IS-luck) is forced to have OOS-anti-luck → low OOS rank → λ < 0.
For iid noise, PBO empirically lands at ~0.7-1.0. Real skill is what
pulls PBO toward 0.

Practical thresholds (calibrated to the complementary-split null):
  PBO < 0.20 → strong evidence the selection process picks real winners
  PBO < 0.50 → good (selection beats the noise baseline materially)
  PBO < 0.70 → mild (selection has some predictive power)
  PBO ≥ 0.70 → indistinguishable from luck — likely overfit

Algorithm (BBLZ 2014, MLFAM Ch.8)
---------------------------------
1. Split T return-period rows into S even-size chunks.
2. For each C(S, S/2) combination:
   a. Form IS = union of S/2 chunks, OOS = complementary S/2 chunks
   b. Compute per-strategy Sharpe on IS and OOS
   c. Pick n* = argmax IS-Sharpe
   d. Rank n* among the M strategies on OOS-Sharpe (1=lowest, M=highest)
   e. Normalize ω_c = rank / (M + 1), logit λ_c = log(ω_c / (1-ω_c))
3. PBO = fraction of combinations with λ_c < 0 (i.e. n* below median OOS)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CSCVResult:
    pbo: float                # Probability of Backtest Overfitting
    logits: np.ndarray        # logit per combination (sign tells overfit dir)
    n_combinations: int
    n_strategies: int         # M
    n_chunks: int             # S
    rows_per_chunk: int


def _sharpe_columnwise(R: np.ndarray) -> np.ndarray:
    """Per-column annualization-free Sharpe (mean / std). Handles zero-var
    columns by returning -inf for those (so argmax never picks them)."""
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    out = np.where(sd > 0, mu / sd, -np.inf)
    return out


def cscv(
    returns_matrix: np.ndarray,
    n_chunks: int = 16,
) -> CSCVResult:
    """Compute PBO for an M-strategy return matrix.

    returns_matrix : array of shape (T, M) — rows are periods, columns
                     are competing strategy variants.
    n_chunks (S)   : number of chunks to split T into. Must be even and
                     small enough that C(S, S/2) is tractable. Default 16
                     yields 12,870 combinations.

    Returns CSCVResult with pbo ∈ [0, 1].
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got shape {R.shape}")
    T, M = R.shape
    if M < 2:
        raise ValueError(f"need at least 2 strategies, got M={M}")
    if n_chunks < 4 or n_chunks % 2 != 0:
        raise ValueError(f"n_chunks must be even and >= 4, got {n_chunks}")
    S = n_chunks
    rows_per_chunk = T // S
    if rows_per_chunk < 2:
        raise ValueError(
            f"T={T} too small for S={S} chunks (got rows_per_chunk={rows_per_chunk})"
        )
    truncated = rows_per_chunk * S
    R = R[:truncated]
    chunks = R.reshape(S, rows_per_chunk, M)

    logits = []
    chunk_idx = list(range(S))
    half = S // 2
    for is_combo in itertools.combinations(chunk_idx, half):
        is_set = set(is_combo)
        oos_combo = [c for c in chunk_idx if c not in is_set]
        R_is = np.vstack([chunks[i] for i in is_combo])
        R_oos = np.vstack([chunks[i] for i in oos_combo])
        s_is = _sharpe_columnwise(R_is)
        s_oos = _sharpe_columnwise(R_oos)
        if not np.isfinite(s_is).any():
            continue
        n_star = int(np.argmax(s_is))
        # OOS rank of n_star: 1 = worst OOS, M = best OOS
        rank = 1 + int((s_oos < s_oos[n_star]).sum())
        omega = rank / (M + 1)
        logit = math.log(omega / (1.0 - omega))
        logits.append(logit)

    logits_arr = np.array(logits)
    pbo = float((logits_arr < 0.0).mean()) if len(logits_arr) else float("nan")
    return CSCVResult(
        pbo=pbo,
        logits=logits_arr,
        n_combinations=len(logits_arr),
        n_strategies=M,
        n_chunks=S,
        rows_per_chunk=rows_per_chunk,
    )


def pbo_interpretation(pbo: float) -> str:
    """Thresholds calibrated against the complementary-split null
    (iid noise → PBO ≈ 0.7-1.0, NOT 0.5)."""
    if pbo < 0.20:
        return "STRONG (selection process has clear predictive power)"
    if pbo < 0.50:
        return "GOOD (selection beats the complementary-split null materially)"
    if pbo < 0.70:
        return "MILD (selection has some predictive power)"
    return "OVERFIT (PBO indistinguishable from iid-noise baseline)"
