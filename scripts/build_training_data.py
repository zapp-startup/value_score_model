#!/usr/bin/env python3
"""
Build train.py-ready CSVs in data/ from Django-style exports:

  subscriptions_merchant.csv   -> merchants.csv
  transactions_transaction.csv -> transactions.csv
  users_userrawexplicit.csv    -> user_explicit.csv
  users_usercomputed.csv       -> user_computed.csv
  users_userrawinferred.csv    -> user_inferred.csv
  subscriptions_subscription1_only_utilization_costbenefit_fixed.csv (if present) or
  subscriptions.csv / subscriptions_subscription.csv -> subscriptions.csv (normalized enums)
  ai_userfact.csv              -> user_facts.csv (parsed numeric fact_value)

Also writes feedback_signals.csv from subscription-level feedback.

Run from repo root:  python scripts/build_training_data.py
"""

from __future__ import annotations

import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_fact_value(raw: object) -> float | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s.replace("null", "None"))
    except (SyntaxError, ValueError, TypeError):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, dict):
        for k in ("preference", "score", "value", "fact_value"):
            if k in v and isinstance(v[k], (int, float)):
                return float(v[k])
    return None


def normalize_subscription_status(s: pd.Series) -> pd.Series:
    m = s.astype(str).str.strip().str.lower()
    m = m.replace({"cancelled": "canceled"})
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    args = parser.parse_args()
    d = args.data_dir

    # --- merchants ---
    merch = pd.read_csv(d / "subscriptions_merchant.csv")
    merchants = merch[["id", "name", "category"]].copy()
    merchants.to_csv(d / "merchants.csv", index=False)
    print(f"merchants.csv: {len(merchants)} rows")

    # --- transactions ---
    tx = pd.read_csv(d / "transactions_transaction.csv")
    # Required for feature pipeline
    need = [
        "id", "user_id", "merchant_id", "direction", "amount", "occurred_at",
        "satisfaction_rating", "regret_rating", "repurchase_likelihood",
        "usage_frequency", "impulse_score", "regret_score",
    ]
    missing = [c for c in need if c not in tx.columns]
    if missing:
        raise SystemExit(f"transactions missing columns: {missing}")
    tx.to_csv(d / "transactions.csv", index=False)
    print(f"transactions.csv: {len(tx)} rows")

    # --- users ---
    ue = pd.read_csv(d / "users_userrawexplicit.csv")
    ue.to_csv(d / "user_explicit.csv", index=False)
    print(f"user_explicit.csv: {len(ue)} rows")

    uc = pd.read_csv(d / "users_usercomputed.csv")
    uc.to_csv(d / "user_computed.csv", index=False)
    print(f"user_computed.csv: {len(uc)} rows")

    ui = pd.read_csv(d / "users_userrawinferred.csv")
    ui.to_csv(d / "user_inferred.csv", index=False)
    print(f"user_inferred.csv: {len(ui)} rows")

    # --- subscriptions (prefer fixed util/cost_benefit export when present) ---
    fixed_sub = d / "subscriptions_subscription1_only_utilization_costbenefit_fixed.csv"
    sub_path = d / "subscriptions.csv"
    if fixed_sub.exists():
        sub_path = fixed_sub
        print(f"  (using {fixed_sub.name} for subscription_utilization / subscription_cost_benefit)")
    elif not sub_path.exists():
        sub_path = d / "subscriptions_subscription.csv"
    subs = pd.read_csv(sub_path)
    if "status" in subs.columns:
        subs["status"] = normalize_subscription_status(subs["status"])
    subs.to_csv(d / "subscriptions.csv", index=False)
    print(f"subscriptions.csv: {len(subs)} rows")

    # --- feedback_signals from subscription table ---
    fb_cols = ["user_id", "merchant_id", "feedback_value_score", "feedback_confidence"]
    if all(c in subs.columns for c in fb_cols):
        fb = subs[fb_cols].drop_duplicates(subset=["user_id", "merchant_id"], keep="last")
        fb.to_csv(d / "feedback_signals.csv", index=False)
        print(f"feedback_signals.csv: {len(fb)} rows")
    else:
        print("feedback_signals.csv: skipped (columns missing on subscriptions)")

    # --- user_facts from AI export ---
    facts_path = d / "ai_userfact.csv"
    if facts_path.exists():
        af = pd.read_csv(facts_path)
        if "fact_value_json" in af.columns and "fact_key" in af.columns:
            vals = af["fact_value_json"].map(_parse_fact_value)
            uf = af.assign(fact_value=vals)
            uf = uf[uf["fact_value"].notna()].copy()
            # Drop non-numeric / junk keys if needed
            uf = uf[uf["fact_key"].notna()]
            out_cols = ["user_id", "fact_key", "fact_value", "confidence"]
            if not all(c in uf.columns for c in out_cols):
                print("user_facts.csv: unexpected ai_userfact schema, skipping")
            else:
                uf[out_cols].to_csv(d / "user_facts.csv", index=False)
                print(f"user_facts.csv: {len(uf)} rows (parsed fact_value)")
        else:
            print("user_facts.csv: skipped (no fact_value_json)")
    else:
        print("user_facts.csv: skipped (no ai_userfact.csv)")


if __name__ == "__main__":
    main()
