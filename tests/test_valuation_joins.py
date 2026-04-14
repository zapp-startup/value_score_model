"""Tests for valuation export schema and transaction_id join integrity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from value_score_model.data.valuation_joins import (
    assert_item_valuation_schema,
    assert_subscription_valuation_schema,
    assert_transaction_valuation_schema,
    merge_transaction_valuations_left,
    orphan_transaction_valuation_ids,
    transaction_valuation_join_stats,
)


def test_assert_transaction_valuation_schema_ok():
    df = pd.DataFrame(
        {"transaction_id": [1, 2], "user_id": [10, 10], "value_score": [80, 90]}
    )
    assert_transaction_valuation_schema(df)


def test_assert_transaction_valuation_schema_missing_column():
    df = pd.DataFrame({"transaction_id": [1], "user_id": [1]})
    with pytest.raises(ValueError, match="value_score"):
        assert_transaction_valuation_schema(df)


def test_transaction_valuation_join_stats_full_match():
    tx = pd.DataFrame({"id": [1, 2, 3]})
    val = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "user_id": [9, 9],
            "value_score": [70, 71],
        }
    )
    s = transaction_valuation_join_stats(tx, val)
    assert s["n_matched_valuation_transaction_ids"] == 2
    assert s["n_orphan_valuation_transaction_ids"] == 0
    assert s["orphan_valuation_transaction_ids"] == []
    assert s["pct_valuation_txn_ids_found_in_transactions"] == 1.0


def test_transaction_valuation_join_stats_orphans():
    tx = pd.DataFrame({"id": [1, 2]})
    val = pd.DataFrame(
        {
            "transaction_id": [1, 99, 100],
            "user_id": [1, 1, 1],
            "value_score": [50, 51, 52],
        }
    )
    s = transaction_valuation_join_stats(tx, val)
    assert s["n_orphan_valuation_transaction_ids"] == 2
    assert set(s["orphan_valuation_transaction_ids"]) == {99, 100}
    orphans = orphan_transaction_valuation_ids(tx, val)
    assert sorted(orphans) == [99, 100]


def test_transaction_valuation_join_stats_duplicate_valuation_rows():
    tx = pd.DataFrame({"id": [1]})
    val = pd.DataFrame(
        {
            "transaction_id": [1, 1],
            "user_id": [5, 5],
            "value_score": [60, 65],
        }
    )
    s = transaction_valuation_join_stats(tx, val)
    assert s["n_valuation_rows"] == 2
    assert s["n_matched_valuation_transaction_ids"] == 1


def test_merge_transaction_valuations_left():
    tx = pd.DataFrame({"id": [1, 2], "amount": [10.0, 20.0]})
    val = pd.DataFrame(
        {
            "transaction_id": [1],
            "user_id": [7],
            "value_score": [88],
            "confidence": [0.9],
        }
    )
    m = merge_transaction_valuations_left(tx, val)
    assert len(m) == 2
    assert m.loc[m["id"] == 1, "value_score"].iloc[0] == 88
    assert pd.isna(m.loc[m["id"] == 2, "value_score"].iloc[0])


def test_assert_subscription_and_item_schemas():
    assert_subscription_valuation_schema(
        pd.DataFrame({"subscription_id": [1], "user_id": [2]})
    )
    assert_item_valuation_schema(
        pd.DataFrame({"id": [1], "user_id": [2], "personal_value_score": [100]})
    )


def test_build_training_data_script_smoke(tmp_path):
    """Minimal Django-style export dir → normalized CSVs including valuations."""
    d = tmp_path / "export"
    d.mkdir()
    pd.DataFrame(
        [
            {"id": 1, "name": "Acme", "category": "software"},
        ]
    ).to_csv(d / "subscriptions_merchant.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": 10,
                "user_id": 1,
                "merchant_id": 1,
                "subscription_id": "",
                "direction": "spend",
                "amount": "9.99",
                "currency": "USD",
                "occurred_at": "2024-01-01T00:00:00Z",
                "category": "food",
                "payment_channel": "card",
                "description_raw": "",
                "satisfaction_rating": "8",
                "regret_rating": "20",
                "repurchase_likelihood": "70",
                "usage_frequency": "1",
                "impulse_score": "0",
                "regret_score": "0",
            }
        ]
    ).to_csv(d / "transactions_transaction.csv", index=False)
    pd.DataFrame([{"user_id": 1, "display_name": "u", "age_range": "25-34", "location_zip": "00000", "life_stage": "other", "employment_type": "ft", "income_range": "50k-75k", "monthly_income": "5000", "financial_goal": "save_more", "risk_tolerance": "medium", "budget_style": "flexible", "created_at": "2024-01-01", "updated_at": "2024-01-01"}]).to_csv(
        d / "users_userrawexplicit.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "user_id": 1,
                "spending_personality": "x",
                "product_spending_style": "x",
                "subscription_behavior_type": "x",
                "cost_weight": 0.33,
                "quality_weight": 0.33,
                "sustainability_weight": 0.34,
                "impulse_susceptibility_score": 0.5,
                "regret_sensitivity": 0.5,
                "budget_adherence_score": 0.5,
                "updated_at": "2024-01-01",
                "feature_logic_version": "v1",
            }
        ]
    ).to_csv(d / "users_usercomputed.csv", index=False)
    pd.DataFrame(
        [
            {
                "user_id": 1,
                "window_days": 30,
                "category_distribution_json": "{}",
                "subscription_usage_frequency_json": "{}",
                "sparse_signals_json": "{}",
                "computed_at": "2024-01-01",
                "data_sufficiency_tier": "high",
                "feature_logic_version": "v1",
            }
        ]
    ).to_csv(d / "users_userrawinferred.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": 100,
                "plan_name": "p",
                "status": "active",
                "billing_cycle": "monthly",
                "price": "10",
                "currency": "USD",
                "started_on": "2023-01-01",
                "merchant_id": 1,
                "user_id": 1,
                "reactivation_count": "0",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            }
        ]
    ).to_csv(d / "subscriptions_subscription.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": 500,
                "context": "one_off_purchase",
                "value_score": 120,
                "base_value_score": 100,
                "confidence": 0.85,
                "tier_used": "2",
                "inference_status": "ok",
                "evidence_json": "{}",
                "reasoning_json": "{}",
                "created_at": "2024-01-02",
                "model_version_id": 1,
                "transaction_id": 10,
                "user_id": 1,
            }
        ]
    ).to_csv(d / "valuations_transactionvaluation.csv", index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_training_data.py"
    r = subprocess.run(
        [sys.executable, str(script), "--data-dir", str(d)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out_tx = pd.read_csv(d / "transactions.csv")
    out_tv = pd.read_csv(d / "transaction_valuations.csv")
    stats = transaction_valuation_join_stats(out_tx, out_tv)
    assert stats["n_matched_valuation_transaction_ids"] == 1
    assert stats["n_orphan_valuation_transaction_ids"] == 0
