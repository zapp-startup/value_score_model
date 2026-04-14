"""
Overutilisation bonus: extra score mass (0–bonus_max) on top of base 0–100.

Only applied when usage clearly exceeds what the billing cycle implies as a
baseline, with recency and sustained-engagement terms — not a flat scale factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# billing_cycle_id from feature_engineering._encode_categorical(BILLING_CYCLE_VOCAB)
# 0 = unknown, 1 = weekly, 2 = monthly, 3 = yearly, 4 = other
_EXPECTED_TXNS_30D = np.array([1.0, 30.0 / 7.0, 1.0, 1.0 / 12.0, 1.0], dtype=np.float64)


def compute_overutilisation_bonus(meta: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Return shape (n,) bonus in [0, bonus_max]. Zero when not overutilising.

    Combines:
      - utilization ratio: actual 30d txns vs expected from billing cycle
      - usage frequency vs a neutral baseline
      - usage_decay (recent engagement)
      - lifetime velocity: total txns vs expected for subscription age
    Gated so ratio must exceed min_ratio and activity must be recent enough.
    """
    cfg = config.get("overutilisation", {})
    bonus_max = float(cfg.get("bonus_max", 50))
    min_ratio = float(cfg.get("min_ratio", 1.12))
    ratio_cap = float(cfg.get("ratio_full_saturation", 3.0))
    min_txns_30d = int(cfg.get("min_transactions_30d", 2))
    min_usage_decay = float(cfg.get("min_usage_decay", 0.35))

    n = len(meta)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    required = ("transaction_count_30d", "billing_cycle_id", "usage_decay")
    if not all(c in meta.columns for c in required):
        return np.zeros(n, dtype=np.float32)

    txn_30 = meta["transaction_count_30d"].fillna(0).to_numpy(dtype=np.float64)
    bc = meta["billing_cycle_id"].fillna(0).to_numpy(dtype=np.int64)
    bc = np.clip(bc, 0, len(_EXPECTED_TXNS_30D) - 1)
    expected = _EXPECTED_TXNS_30D[bc]
    expected = np.maximum(expected, 0.12)

    ratio = txn_30 / expected

    usage_freq = meta.get(
        "usage_frequency", pd.Series(1.0, index=meta.index)
    ).fillna(1.0).to_numpy(dtype=np.float64)

    usage_decay = meta["usage_decay"].fillna(0).to_numpy(dtype=np.float64)

    days_last = meta.get(
        "days_since_last_used", pd.Series(999.0, index=meta.index)
    ).fillna(999.0).to_numpy(dtype=np.float64)

    total_txns = meta.get(
        "total_transaction_count", pd.Series(0.0, index=meta.index)
    ).fillna(0).to_numpy(dtype=np.float64)

    days_started = meta.get(
        "days_since_started", pd.Series(180.0, index=meta.index)
    ).fillna(180.0).to_numpy(dtype=np.float64)

    months_active = np.maximum(days_started / 30.0, 0.5)
    expected_lifetime = expected * months_active
    velocity_ratio = total_txns / np.maximum(expected_lifetime, 0.5)
    velocity_score = np.clip((velocity_ratio - 1.0) / 2.0, 0.0, 1.0)

    util_excess = (ratio - 1.0) / max(ratio_cap - 1.0, 0.25)
    util_score = np.clip(util_excess, 0.0, 1.0)

    freq_score = np.clip((usage_freq - 1.0) / 4.0, 0.0, 1.0)

    recency_w = float(cfg.get("recency_half_life_days", 18.0))
    recency_penalty = np.exp(-np.maximum(0.0, days_last - 10.0) / recency_w)

    w_util = float(cfg.get("weight_utilization", 0.42))
    w_freq = float(cfg.get("weight_frequency", 0.22))
    w_decay = float(cfg.get("weight_decay", 0.21))
    w_vel = float(cfg.get("weight_velocity", 0.15))

    composite = (
        w_util * util_score
        + w_freq * freq_score
        + w_decay * np.clip(usage_decay, 0.0, 1.0)
        + w_vel * velocity_score
    )
    nonlin = float(cfg.get("composite_exponent", 1.18))
    shaped = np.power(np.clip(composite, 0.0, 1.0), nonlin)

    gate_ratio = (ratio >= min_ratio).astype(np.float64)
    gate_txn = (txn_30 >= min_txns_30d).astype(np.float64)
    gate_decay = (usage_decay >= min_usage_decay).astype(np.float64)

    bonus = (
        bonus_max
        * shaped
        * recency_penalty
        * gate_ratio
        * gate_txn
        * gate_decay
    )
    return np.clip(bonus, 0.0, bonus_max).astype(np.float32)


def apply_overutilisation_guardrails(
    base_scores: np.ndarray,
    bonus: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Post-process raw bonus using learned base (0–learned_max). Zeros bonus when base is
    below min_base_score_for_any_bonus. Optional bonus_ramp_top_base: linear ramp of the
    bonus multiplier from 0 at min_base to 1 at ramp_top (only applies where base >= min_base).
    """
    cfg = config.get("overutilisation", {})
    bonus_max = float(cfg.get("bonus_max", 50))
    out = np.asarray(bonus, dtype=np.float32).copy()
    base_scores = np.asarray(base_scores, dtype=np.float64)

    min_any = float(cfg.get("min_base_score_for_any_bonus", 0))
    ramp_top = float(cfg.get("bonus_ramp_top_base", 0))

    if min_any > 0:
        out = np.where(base_scores < min_any, 0.0, out)
        if ramp_top > min_any:
            mult = np.clip((base_scores - min_any) / (ramp_top - min_any), 0.0, 1.0)
            out = np.where(base_scores < min_any, 0.0, out * mult.astype(np.float32))
        return np.clip(out, 0.0, bonus_max).astype(np.float32)

    if ramp_top > 0:
        mult = np.clip(base_scores / ramp_top, 0.0, 1.0)
        out = (out * mult.astype(np.float32))
    return np.clip(out, 0.0, bonus_max).astype(np.float32)
