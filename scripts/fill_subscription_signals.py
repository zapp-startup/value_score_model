#!/usr/bin/env python3
"""
Fill subscription_utilization and subscription_cost_benefit on subscription CSVs.

Uses only columns present in the export (usage_frequency, feedback, status, price)
plus merchant name/category — not ground-truth watch time or delivery savings.
Values are deterministic (same row id → same numbers) for reproducible training.

Usage (from repo root):
  python scripts/fill_subscription_signals.py
  python scripts/fill_subscription_signals.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(0.0, 1.0)


def _status_multiplier(status: pd.Series) -> pd.Series:
    s = status.fillna("active").str.lower()
    return s.map(lambda x: 0.62 if x == "canceled" else (0.82 if x == "paused" else 1.0))


def _cost_benefit_base(name: str, category: str) -> float:
    n = (name or "").lower()
    c = (category or "other").lower()
    if "amazon prime" in n:
        return 0.44
    if "costco" in n:
        return 0.38
    if "doordash" in n or "uber eats" in n or "grubhub" in n:
        return 0.36
    if c == "streaming":
        return 0.09
    if c == "fitness":
        return 0.12
    if c == "software":
        return 0.14
    if c == "utilities":
        return 0.17
    if c == "education":
        return 0.10
    if c == "grocery":
        return 0.21
    if c == "food":
        return 0.18
    return 0.11


def compute_signals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (subscription_utilization, subscription_cost_benefit)."""
    uf = pd.to_numeric(df["usage_frequency"], errors="coerce")
    uf = uf.fillna(uf.median())
    # Cap using high percentile so outliers don't dominate
    cap = float(uf.quantile(0.99)) if len(uf) else 14.0
    cap = max(cap, 7.0)
    usage_norm = (uf / cap).clip(0, 1)

    fv = pd.to_numeric(df["feedback_value_score"], errors="coerce").fillna(0.55)
    fc = pd.to_numeric(df["feedback_confidence"], errors="coerce").fillna(0.5)
    blended_feedback = fv * fc + 0.5 * (1.0 - fc)

    engagement = 0.52 * usage_norm + 0.48 * blended_feedback
    util = engagement * _status_multiplier(df.get("status", pd.Series("active", index=df.index)))

    # Deterministic row-level jitter (reproducible, vectorized)
    ids = df["id"].astype(np.float64).to_numpy()
    jitter_u = (np.sin(ids * 0.017) + np.cos(ids * 0.003)) * 0.02
    subscription_utilization = _clip01(util + jitter_u)

    bases = df.apply(
        lambda r: _cost_benefit_base(str(r.get("merchant_name", "")), str(r.get("merchant_category", ""))),
        axis=1,
    )
    jitter_cb = (np.sin(ids * 0.031) + np.cos(ids * 0.011)) * 0.025
    # Weak coupling: usage + feedback nudge cost_benefit; base is merchant-driven
    cb = bases + 0.22 * usage_norm + 0.12 * (fv - 0.5) + jitter_cb
    subscription_cost_benefit = _clip01(pd.Series(cb.values, index=df.index))

    return subscription_utilization, subscription_cost_benefit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    subs_path = data_dir / "subscriptions_subscription.csv"
    merch_path = data_dir / "subscriptions_merchant.csv"

    if not subs_path.exists():
        raise SystemExit(f"Missing {subs_path}")
    if not merch_path.exists():
        raise SystemExit(f"Missing {merch_path}")

    subs = pd.read_csv(subs_path)
    merch = pd.read_csv(merch_path)
    m = merch[["id", "name", "category"]].rename(
        columns={"id": "merchant_id", "name": "merchant_name", "category": "merchant_category"}
    )
    df = subs.merge(m, on="merchant_id", how="left")
    df["merchant_name"] = df["merchant_name"].fillna("")
    df["merchant_category"] = df["merchant_category"].fillna("other")

    su, scb = compute_signals(df)

    out = subs.copy()
    out["subscription_utilization"] = su.values.round(6)
    out["subscription_cost_benefit"] = scb.values.round(6)

    if args.dry_run:
        print(out[["id", "merchant_id", "subscription_utilization", "subscription_cost_benefit"]].head(10))
        print("...")
        print(
            out[["subscription_utilization", "subscription_cost_benefit"]].describe().to_string()
        )
        return

    out.to_csv(subs_path, index=False)
    # Train script expects subscriptions.csv in data dir
    out.to_csv(data_dir / "subscriptions.csv", index=False)
    print(f"Wrote {len(out)} rows to {subs_path}")
    print(f"Wrote {len(out)} rows to {data_dir / 'subscriptions.csv'}")


if __name__ == "__main__":
    main()
