"""Module B: build candidate INTERSECTION gates from attribution table."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import pandas as pd


def select_top_k_significant(
    table: pd.DataFrame,
    signal: str,
    k: int = 3,
) -> list[dict]:
    """For a given signal, return top-K significant attribution rows by spread.

    Spread = max bucket mean - min bucket mean for that channel.
    Returns list of dicts: {channel, bucket, mean_bps, spread_bps}.
    """
    sig_rows = table[(table["signal"] == signal) & (table["significant"])]
    if sig_rows.empty:
        return []
    # Compute per-channel spread = max bucket mean − min bucket mean
    chan_spreads = {}
    for ch in sig_rows["channel"].unique():
        ch_buckets = table[(table["signal"] == signal) & (table["channel"] == ch)]
        chan_spreads[ch] = float(ch_buckets["mean_bps"].max() -
                                  ch_buckets["mean_bps"].min())
    # Sort channels by spread descending, take top-K that have a significant row
    sorted_channels = sorted(chan_spreads.items(),
                             key=lambda kv: kv[1], reverse=True)
    chosen = []
    for ch, spread in sorted_channels[:k]:
        # Pick the bucket that is significant (there should be exactly one
        # significant row per channel; if multiple, take the one with max mean)
        ch_sig = sig_rows[sig_rows["channel"] == ch].sort_values(
            "mean_bps", ascending=False)
        if ch_sig.empty:
            continue
        row = ch_sig.iloc[0]
        chosen.append({
            "channel": str(row["channel"]),
            "bucket": str(row["bucket"]),
            "mean_bps": float(row["mean_bps"]),
            "spread_bps": spread,
        })
    return chosen


GateType = Literal["ALWAYS_ON", "INTERSECTION", "NO_GATE_FOUND"]


@dataclass(frozen=True)
class GateSpec:
    signal: str
    gate_type: GateType
    channels: list[dict]                  # [{channel, bucket, mean_bps, spread_bps}]
    expected_fire_rate_pct: float
    reason: str


def build_candidate_gates(
    table: pd.DataFrame,
    baseline_signal: str,
    k: int = 3,
) -> list[GateSpec]:
    """For each signal in the attribution table, emit a GateSpec.

    baseline_signal gets ALWAYS_ON (no gate needed — it's the always-on baseline).
    Other signals get INTERSECTION (top-K) or NO_GATE_FOUND.

    expected_fire_rate_pct is approximated as (1/3)^k for INTERSECTION
    (each tercile gate fires ~33%; assumed independent).
    """
    out: list[GateSpec] = []
    # Always emit baseline (ALWAYS_ON) even if not present in the attribution table
    out.append(GateSpec(
        signal=baseline_signal, gate_type="ALWAYS_ON", channels=[],
        expected_fire_rate_pct=100.0,
        reason="baseline ranker, no gate needed",
    ))
    for signal in sorted(table["signal"].unique()):
        if signal == baseline_signal:
            continue  # already emitted above
        chosen = select_top_k_significant(table, signal=signal, k=k)
        if not chosen:
            out.append(GateSpec(
                signal=signal, gate_type="NO_GATE_FOUND", channels=[],
                expected_fire_rate_pct=0.0,
                reason="no channel significant after bootstrap CI",
            ))
        else:
            fire_rate = (1.0 / 3.0) ** len(chosen) * 100
            out.append(GateSpec(
                signal=signal, gate_type="INTERSECTION", channels=chosen,
                expected_fire_rate_pct=round(fire_rate, 1),
                reason=f"{len(chosen)} channels with significant lift",
            ))
    return out


def save_candidate_gates(gates: list[GateSpec], out_path: Path) -> None:
    payload = {g.signal: asdict(g) for g in gates}
    out_path.write_text(json.dumps(payload, indent=2, default=str))


def load_candidate_gates(in_path: Path) -> list[GateSpec]:
    raw = json.loads(in_path.read_text())
    return [GateSpec(**v) for v in raw.values()]
