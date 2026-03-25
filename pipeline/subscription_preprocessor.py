"""
Map subscription-specific raw metrics (watch time, listen time, savings, etc.)
into two generic model features: subscription_utilization and subscription_cost_benefit.

Reads optional `subscription_usage` (per subscription_id) or the same column names on
`subscriptions`. Profiles match merchant `name` (case-insensitive substring).
See configs/default.yaml → subscription_signals.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

SIGNAL_UTILIZATION = "subscription_utilization"
SIGNAL_COST_BENEFIT = "subscription_cost_benefit"
SIGNAL_COLUMNS = (SIGNAL_UTILIZATION, SIGNAL_COST_BENEFIT)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _get_value(row: pd.Series, column: str) -> Optional[float]:
    if column not in row.index:
        return None
    v = row[column]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _select_profile(merchant_name_lower: str, profiles: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """First non-wildcard substring match wins; otherwise profile with match '*'."""
    specific = [p for p in profiles if str(p.get("match", "*")).strip() != "*"]
    wild = [p for p in profiles if str(p.get("match", "*")).strip() == "*"]
    for p in specific:
        m = str(p.get("match", "")).lower()
        if m and m in merchant_name_lower:
            return p
    if wild:
        return wild[-1]
    return None


def apply_subscription_preprocessor(
    data: dict[str, pd.DataFrame],
    config: dict,
) -> dict[str, pd.DataFrame]:
    """
    Return a copy of `data` with `subscriptions` containing
    subscription_utilization and subscription_cost_benefit in [0, 1].
    """
    cfg = config.get("subscription_signals", {})
    out = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in data.items()}
    subs = out["subscriptions"].copy()
    default_u = float(cfg.get("default_utilization", 0.5))
    default_cb = float(cfg.get("default_cost_benefit", 0.0))
    override = bool(cfg.get("override_existing", False))
    profiles: list[dict[str, Any]] = list(cfg.get("profiles", []))

    if not cfg.get("enabled", True):
        subs = _apply_defaults(subs, default_u, default_cb)
        out["subscriptions"] = subs
        return out

    merchants = out["merchants"]
    if "name" not in merchants.columns:
        subs = _apply_defaults(subs, default_u, default_cb)
        out["subscriptions"] = subs
        return out

    m = merchants[["id", "name"]].rename(columns={"id": "merchant_id"})
    enriched = subs.merge(m, on="merchant_id", how="left")
    enriched["merchant_name_lower"] = enriched["name"].fillna("").str.lower()

    usage = out.get("subscription_usage")
    if usage is not None and not usage.empty and "subscription_id" in usage.columns:
        u = usage.copy()
        overlap = [c for c in u.columns if c != "subscription_id" and c in enriched.columns]
        if overlap:
            u = u.drop(columns=overlap, errors="ignore")
        enriched = enriched.merge(
            u,
            left_on="id",
            right_on="subscription_id",
            how="left",
            suffixes=("", "_usage"),
        )

    price = enriched.get("price", pd.Series(0.0, index=enriched.index)).astype(float).replace(0, np.nan)
    price_filled = price.fillna(1.0)

    util = np.zeros(len(enriched), dtype=np.float64)
    cb = np.zeros(len(enriched), dtype=np.float64)

    for i in range(len(enriched)):
        row = enriched.iloc[i]
        pr = float(price_filled.iloc[i])
        mname = str(row.get("merchant_name_lower", "") or "")
        prof = _select_profile(mname, profiles)

        # --- utilization ---
        if override and SIGNAL_UTILIZATION in row.index and pd.notna(row.get(SIGNAL_UTILIZATION)):
            util[i] = _clip01(float(row[SIGNAL_UTILIZATION]))
        else:
            computed_u = _profile_utilization(row, prof) if prof else None
            if computed_u is not None:
                util[i] = computed_u
            elif SIGNAL_UTILIZATION in row.index and pd.notna(row.get(SIGNAL_UTILIZATION)):
                util[i] = _clip01(float(row[SIGNAL_UTILIZATION]))
            else:
                util[i] = default_u

        # --- cost benefit ---
        if override and SIGNAL_COST_BENEFIT in row.index and pd.notna(row.get(SIGNAL_COST_BENEFIT)):
            cb[i] = _clip01(float(row[SIGNAL_COST_BENEFIT]))
        else:
            computed_cb = _profile_cost_benefit(row, prof, pr) if prof else None
            if computed_cb is not None:
                cb[i] = computed_cb
            elif SIGNAL_COST_BENEFIT in row.index and pd.notna(row.get(SIGNAL_COST_BENEFIT)):
                cb[i] = _clip01(float(row[SIGNAL_COST_BENEFIT]))
            else:
                cb[i] = default_cb

    subs[SIGNAL_UTILIZATION] = util
    subs[SIGNAL_COST_BENEFIT] = cb
    out["subscriptions"] = subs
    return out


def _apply_defaults(subs: DataFrame, u: float, cb: float) -> DataFrame:
    if SIGNAL_UTILIZATION not in subs.columns:
        subs[SIGNAL_UTILIZATION] = u
    else:
        subs[SIGNAL_UTILIZATION] = subs[SIGNAL_UTILIZATION].fillna(u)
    if SIGNAL_COST_BENEFIT not in subs.columns:
        subs[SIGNAL_COST_BENEFIT] = cb
    else:
        subs[SIGNAL_COST_BENEFIT] = subs[SIGNAL_COST_BENEFIT].fillna(cb)
    subs[SIGNAL_UTILIZATION] = subs[SIGNAL_UTILIZATION].astype(float).clip(0, 1)
    subs[SIGNAL_COST_BENEFIT] = subs[SIGNAL_COST_BENEFIT].astype(float).clip(0, 1)
    return subs


def _profile_utilization(row: pd.Series, prof: Optional[dict[str, Any]]) -> Optional[float]:
    if not prof:
        return None
    parts: list[float] = []
    for spec in prof.get("metrics", []):
        if spec.get("driver", "utilization") != "utilization":
            continue
        col = spec.get("column")
        if not col:
            continue
        raw = _get_value(row, col)
        if raw is None:
            continue
        baseline = float(spec.get("baseline", 1.0))
        if baseline <= 0:
            continue
        parts.append(_clip01(raw / baseline))
    if not parts:
        return None
    return float(np.mean(parts))


def _profile_cost_benefit(row: pd.Series, prof: Optional[dict[str, Any]], price: float) -> Optional[float]:
    if not prof:
        return None
    parts: list[float] = []
    for spec in prof.get("metrics", []):
        if spec.get("driver", "utilization") != "cost_benefit":
            continue
        col = spec.get("column")
        if not col:
            continue
        raw = _get_value(row, col)
        if raw is None:
            continue
        norm = spec.get("normalize", "price")
        if norm == "price":
            denom = max(price, 1.0)
            parts.append(_clip01(raw / denom))
        else:
            baseline = float(spec.get("baseline", 1.0))
            if baseline <= 0:
                continue
            parts.append(_clip01(raw / baseline))
    if not parts:
        return None
    return float(np.mean(parts))
