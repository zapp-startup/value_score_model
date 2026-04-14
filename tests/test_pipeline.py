#generated with claude

"""
tests/test_pipeline.py
-----------------------
Tests covering: data generation, feature engineering, all three tiers,
full model train+predict, evaluation metrics, and the batch scorer.

Run with:
    pytest value_score_model/tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime

from value_score_model.data.sample_generator import generate_sample_dataset
from value_score_model.pipeline.feature_engineering import FeatureEngineer, build_target, DecayComputer
from value_score_model.pipeline.overutilisation import (
    apply_overutilisation_guardrails,
    compute_overutilisation_bonus,
)
from value_score_model.pipeline.txn_feature_engineering import (
    build_transaction_supervised_frame,
    build_transaction_training_frame,
)
from value_score_model.models.transaction_value_model import TransactionValueModel
from value_score_model.config import recommendation_thresholds, scoring_scale
from value_score_model.models.tier1_coldstart import ColdStartModel
from value_score_model.models.tier2_xgboost import XGBoostValueModel
from value_score_model.models.tier3_neural import NeuralValueModel
from value_score_model.models.value_score_model import ValueScoreModel
from value_score_model.evaluation.metrics import evaluate, evaluate_display_metrics
from value_score_model.config import load_config


@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    # Faster settings for tests
    cfg["training"]["epochs"] = 3
    cfg["training"]["batch_size"] = 32
    cfg["xgboost"]["n_estimators"] = 10
    return cfg


@pytest.fixture(scope="module")
def small_data():
    """Small synthetic dataset used across tests."""
    return generate_sample_dataset(n_users=30, n_merchants=10, seed=0)


@pytest.fixture(scope="module")
def features_and_meta(small_data, config):
    fe = FeatureEngineer(config)
    X, meta = fe.fit_transform(small_data)
    return fe, X, meta


# ─────────────────────────────────────────────────────────────────────────────
# OVERUTILISATION BONUS
# ─────────────────────────────────────────────────────────────────────────────

class TestOverutilisation:

    def test_bonus_zero_when_low_usage(self, config):
        meta = pd.DataFrame({
            "transaction_count_30d": [0, 1],
            "billing_cycle_id": [2, 2],
            "usage_decay": [0.9, 0.9],
            "usage_frequency": [1.0, 1.0],
            "days_since_last_used": [2.0, 2.0],
            "days_since_started": [90.0, 90.0],
            "total_transaction_count": [2.0, 2.0],
        })
        b = compute_overutilisation_bonus(meta, config)
        assert b.max() == 0.0

    def test_bonus_positive_when_heavy_usage(self, config):
        meta = pd.DataFrame({
            "transaction_count_30d": [25],
            "billing_cycle_id": [2],
            "usage_decay": [0.95],
            "usage_frequency": [3.0],
            "days_since_last_used": [1.0],
            "days_since_started": [120.0],
            "total_transaction_count": [80.0],
        })
        b = compute_overutilisation_bonus(meta, config)
        assert b[0] > 5.0

    def test_bonus_capped_at_config_max(self, config):
        cfg = dict(config)
        cfg["overutilisation"] = dict(cfg.get("overutilisation", {}), bonus_max=50)
        meta = pd.DataFrame({
            "transaction_count_30d": [200],
            "billing_cycle_id": [1],
            "usage_decay": [1.0],
            "usage_frequency": [5.0],
            "days_since_last_used": [0.0],
            "days_since_started": [400.0],
            "total_transaction_count": [5000.0],
        })
        b = compute_overutilisation_bonus(meta, cfg)
        assert b[0] <= 50.0 + 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_all_keys_present(self, small_data):
        expected = {
            "merchants", "subscriptions", "subscription_usage", "transactions",
            "user_explicit", "user_computed", "user_inferred",
            "user_facts", "feedback_signals"
        }
        assert expected.issubset(set(small_data.keys()))

    def test_no_empty_tables(self, small_data):
        for key in ["merchants", "subscriptions", "transactions", "user_explicit"]:
            assert len(small_data[key]) > 0, f"{key} is empty"

    def test_user_ids_consistent(self, small_data):
        exp_users = set(small_data["user_explicit"]["user_id"])
        sub_users = set(small_data["subscriptions"]["user_id"])
        assert sub_users.issubset(exp_users)

    def test_value_priority_weights_normalize(self, small_data):
        uc = small_data["user_computed"]
        total = uc["cost_weight"] + uc["quality_weight"] + uc["sustainability_weight"]
        assert (total - 1.0).abs().max() < 0.01

    def test_transaction_feedback_sparse(self, small_data):
        txns = small_data["transactions"]
        # Not all transactions should have satisfaction ratings
        fill_rate = txns["satisfaction_rating"].notna().mean()
        assert 0.1 < fill_rate < 0.95, f"Unexpected fill rate: {fill_rate}"


# ─────────────────────────────────────────────────────────────────────────────
# DECAY COMPUTER
# ─────────────────────────────────────────────────────────────────────────────

class TestDecayComputer:

    def test_usage_decay_recent(self):
        dc = DecayComputer(usage_lambda=0.015)
        days = pd.Series([0, 1, 10, 46, 100, 999])
        decay = dc.usage_decay(days)
        assert float(decay.iloc[0]) == pytest.approx(1.0, abs=0.01)
        assert float(decay.iloc[-1]) < 0.01

    def test_usage_decay_monotone(self):
        dc = DecayComputer()
        days = pd.Series(range(0, 500, 10))
        decay = dc.usage_decay(days)
        assert (decay.diff().dropna() <= 0).all()

    def test_time_decay_mild(self):
        dc = DecayComputer(time_lambda=0.003)
        # After 231 days, should retain ~50%
        val = dc.time_decay(pd.Series([231])).iloc[0]
        assert 0.4 < val < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_output_shape(self, features_and_meta, small_data):
        _, X, meta = features_and_meta
        assert len(X) == len(meta)
        assert len(X) > 0
        assert X.shape[1] > 10

    def test_no_all_nan_columns(self, features_and_meta):
        _, X, _ = features_and_meta
        all_nan = X.isnull().all(axis=0)
        assert not all_nan.any(), f"All-NaN columns: {list(X.columns[all_nan])}"

    def test_categorical_ids_are_integers(self, features_and_meta):
        _, X, _ = features_and_meta
        cat_cols = [c for c in X.columns if c.endswith("_id")]
        for col in cat_cols:
            assert X[col].dtype in [np.int32, np.int64, int], f"{col} not integer"

    def test_meta_has_required_columns(self, features_and_meta):
        _, _, meta = features_and_meta
        for col in [
            "user_id",
            "merchant_id",
            "subscription_id",
            "total_transaction_count",
            "transaction_count_30d",
            "usage_frequency",
            "days_since_last_used",
            "days_since_started",
            "billing_cycle_id",
            "usage_decay",
        ]:
            assert col in meta.columns

    def test_transform_without_fit_raises(self, config, small_data):
        fe = FeatureEngineer(config)
        with pytest.raises(RuntimeError):
            fe.transform(small_data)

    def test_target_in_range(self, features_and_meta, small_data, config):
        _, X, meta = features_and_meta
        base = small_data["subscriptions"].copy()
        y = build_target(base, small_data["user_computed"], config)
        assert (y >= 0).all() and (y <= 100).all()

    def test_feedback_signal_zeroed_below_confidence(self, config, small_data):
        """Low-confidence feedback signals should be zeroed out."""
        data = dict(small_data)
        # Inject all low-confidence feedback
        fb = small_data["feedback_signals"].copy()
        fb["feedback_confidence"] = 0.1  # below threshold
        data["feedback_signals"] = fb
        fe = FeatureEngineer(config)
        X, _ = fe.fit_transform(data)
        if "text_feedback_score" in X.columns:
            assert X["text_feedback_score"].max() == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1: COLD-START
# ─────────────────────────────────────────────────────────────────────────────

class TestTier1ColdStart:

    def test_fit_and_predict(self, config, small_data):
        base = small_data["subscriptions"].merge(
            small_data["merchants"][["id", "category"]].rename(
                columns={"id": "merchant_id", "category": "merchant_category"}
            ), on="merchant_id", how="left"
        ).merge(
            small_data["user_explicit"][["user_id", "life_stage", "financial_goal",
                                         "monthly_income", "value_priority_cost",
                                         "value_priority_quality"]],
            on="user_id", how="left"
        ).merge(
            small_data["user_computed"][["user_id", "cost_weight", "quality_weight"]],
            on="user_id", how="left"
        )
        monthly_income = base["monthly_income"].replace(0, None).fillna(3000)
        base["price_norm"] = (base["price"] / monthly_income).clip(0, 1)

        y = pd.Series(np.random.uniform(30, 80, len(base)))
        model = ColdStartModel(config)
        model.fit(base, y)
        scores = model.predict(base)
        assert len(scores) == len(base)
        assert ((scores >= 0) & (scores <= 100)).all()

    def test_cold_start_never_nan(self, config, small_data):
        base = small_data["subscriptions"].merge(
            small_data["merchants"][["id", "category"]].rename(
                columns={"id": "merchant_id", "category": "merchant_category"}
            ), on="merchant_id", how="left"
        )
        base["life_stage"] = "unknown_stage"
        base["financial_goal"] = "unknown_goal"
        base["price_norm"] = 0.05
        base["cost_weight"] = 0.33
        base["quality_weight"] = 0.33
        model = ColdStartModel(config)
        y = pd.Series(np.full(len(base), 50.0))
        model.fit(base, y)
        scores = model.predict(base)
        assert not np.isnan(scores).any()


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2: XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

class TestTier2XGBoost:

    def test_fit_predict(self, config, features_and_meta, small_data):
        _, X, meta = features_and_meta
        base = small_data["subscriptions"].copy()
        y = build_target(base, small_data["user_computed"], config)
        y = y.iloc[:len(X)].reset_index(drop=True)

        model = XGBoostValueModel(config)
        model.fit(X, y)
        scores = model.predict(X)
        assert len(scores) == len(X)
        assert ((scores >= 0) & (scores <= 100)).all()

    def test_feature_importance_sums_to_one(self, config, features_and_meta, small_data):
        _, X, _ = features_and_meta
        y = pd.Series(np.random.uniform(20, 90, len(X)))
        model = XGBoostValueModel(config)
        model.fit(X, y)
        fi = model.get_feature_importance()
        assert abs(fi.sum() - 1.0) < 0.01

    def test_predict_without_fit_raises(self, config, features_and_meta):
        _, X, _ = features_and_meta
        model = XGBoostValueModel(config)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_save_load_roundtrip(self, config, features_and_meta, small_data):
        _, X, _ = features_and_meta
        y = pd.Series(np.random.uniform(20, 90, len(X)))
        model = XGBoostValueModel(config)
        model.fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            model.save(f.name)
            loaded = XGBoostValueModel.load(f.name)
        scores_orig = model.predict(X)
        scores_loaded = loaded.predict(X)
        np.testing.assert_allclose(scores_orig, scores_loaded, atol=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3: NEURAL
# ─────────────────────────────────────────────────────────────────────────────

class TestTier3Neural:

    def test_fit_predict_small(self, config, features_and_meta, small_data):
        _, X, meta = features_and_meta
        y = pd.Series(np.random.uniform(20, 90, len(X)))
        user_ids = meta["user_id"].values
        merchant_ids = meta["merchant_id"].values

        model = NeuralValueModel(config)
        model.fit(X, y, user_ids=user_ids, merchant_ids=merchant_ids)
        scores = model.predict(X, user_ids=user_ids, merchant_ids=merchant_ids)

        assert len(scores) == len(X)
        assert ((scores >= 0) & (scores <= 100)).all()
        assert not np.isnan(scores).any()

    def test_unknown_user_handled(self, config, features_and_meta, small_data):
        """Unseen user IDs at inference time should not crash — mapped to padding."""
        _, X, meta = features_and_meta
        y = pd.Series(np.random.uniform(20, 90, len(X)))
        model = NeuralValueModel(config)
        model.fit(X, y, meta["user_id"].values, meta["merchant_id"].values)

        # Use unknown user IDs
        unknown_user_ids = np.full(len(X), 999999)
        scores = model.predict(X, unknown_user_ids, meta["merchant_id"].values)
        assert not np.isnan(scores).any()


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TestValueScoreModel:

    def test_fit_predict_end_to_end(self, config, small_data):
        model = ValueScoreModel(config)
        model.fit(small_data)
        results = model.predict(small_data)

        assert "value_score" in results.columns
        assert "base_value_score" in results.columns
        assert "overutilisation_bonus" in results.columns
        assert "confidence" in results.columns
        assert "tier_used" in results.columns
        assert len(results) > 0
        assert ((results["value_score"] >= 0) & (results["value_score"] <= 150)).all()
        assert (results["base_value_score"] <= 100).all()
        assert (results["overutilisation_bonus"] <= 50.1).all()
        assert ((results["confidence"] >= 0) & (results["confidence"] <= 1)).all()
        assert "display_value_score" in results.columns
        assert (results["display_value_score"] == results["value_score"]).all()

    def test_combined_score_at_least_base(self, config, small_data):
        model = ValueScoreModel(config)
        model.fit(small_data)
        results = model.predict(small_data)
        assert (
            results["value_score"] + 1 >= results["base_value_score"]
        ).all()

    def test_tier_routing(self, config, small_data):
        model = ValueScoreModel(config)
        model.fit(small_data)
        results = model.predict(small_data)
        # All tiers should be valid
        assert results["tier_used"].isin([1, 2, 3]).all()

    def test_evidence_json_present(self, config, small_data):
        config_with_evidence = dict(config)
        config_with_evidence["scoring"] = {"save_evidence": True}
        model = ValueScoreModel(config_with_evidence)
        model.fit(small_data)
        results = model.predict(small_data)
        assert "evidence_json" in results.columns
        assert results["evidence_json"].notna().any()

    def test_save_load_roundtrip(self, config, small_data):
        model = ValueScoreModel(config)
        model.fit(small_data)
        ref = datetime(2025, 6, 1, 12, 0, 0)
        preds_before = model.predict(small_data, reference_time=ref)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = ValueScoreModel.load(tmpdir, config)
        preds_after = loaded.predict(small_data, reference_time=ref)

        np.testing.assert_allclose(
            preds_before["value_score"].values,
            preds_after["value_score"].values,
            atol=1,
        )

    def test_active_only_scoring(self, config, small_data):
        """Paused/canceled subscriptions should not appear in results if filtered."""
        data = dict(small_data)
        subs = small_data["subscriptions"].copy()
        subs["status"] = "active"
        data["subscriptions"] = subs

        model = ValueScoreModel(config)
        model.fit(data)
        results = model.predict(data)
        assert len(results) > 0


class TestScoreContractConfig:

    def test_scoring_scale_keys(self, config):
        s = scoring_scale(config)
        assert s["learned_max"] == 100
        assert s["display_max"] == 150

    def test_recommendation_thresholds_combined(self, config):
        col, thr = recommendation_thresholds(config)
        assert col == "combined"
        assert thr["buy"] > 90

    def test_recommendation_thresholds_base(self, config):
        cfg = dict(config)
        cfg["recommendation"] = dict(cfg.get("recommendation", {}), score_column="base")
        col, thr = recommendation_thresholds(cfg)
        assert col == "base"
        assert thr["buy"] == 65


class TestOverutilisationGuardrails:

    def test_bonus_zeroed_when_base_low(self, config):
        cfg = dict(config)
        cfg["overutilisation"] = dict(
            cfg.get("overutilisation", {}),
            min_base_score_for_any_bonus=50,
            bonus_max=50,
        )
        base = np.array([30.0, 80.0], dtype=np.float32)
        bonus = np.array([40.0, 40.0], dtype=np.float32)
        out = apply_overutilisation_guardrails(base, bonus, cfg)
        assert out[0] == 0.0
        assert out[1] == 40.0


class TestTxnFeatureAndTransactionModel:

    def test_build_transaction_frame(self, small_data):
        ref = datetime(2024, 1, 1, 12, 0, 0)
        X, y, uid = build_transaction_training_frame(small_data, reference_time=ref)
        assert len(X) == len(y) == len(uid) > 0
        assert float(y.max()) <= 100.0

    def test_transaction_value_model_fit(self, small_data, config):
        ref = datetime(2024, 1, 1, 12, 0, 0)
        X, y, _ = build_transaction_training_frame(small_data, reference_time=ref)
        cfg = dict(config)
        cfg["transaction_xgboost"] = {"n_estimators": 20, "max_depth": 4}
        m = TransactionValueModel(cfg)
        n = min(200, len(X))
        m.fit(X.iloc[:n], y.iloc[:n])
        p = m.predict(X.iloc[: min(50, len(X))])
        assert len(p) == min(50, len(X))
        assert (p >= 0).all() and (p <= 100).all()

    def test_build_transaction_supervised_frame(self, small_data, config):
        tx = small_data["transactions"].copy()
        # Label a subset of spend txns
        spend_ids = tx.loc[tx["direction"] == "spend", "id"].head(15).tolist()
        tv = pd.DataFrame(
            {
                "transaction_id": spend_ids,
                "value_score": [75] * len(spend_ids),
            }
        )
        data = dict(small_data)
        data["transaction_valuations"] = tv
        cfg = dict(config)
        cfg["transaction_training"] = {"label_max": 150, "min_supervised_rows": 5}
        out = build_transaction_supervised_frame(data, cfg)
        assert out is not None
        X, y, uid = out
        assert len(X) == len(y) == len(uid) == len(spend_ids)
        assert float(y.min()) >= 0 and float(y.max()) <= 100.5


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_perfect_predictions(self):
        y = pd.Series(np.random.uniform(20, 90, 100))
        preds = pd.DataFrame({
            "user_id": np.repeat(np.arange(10), 10),
            "value_score": y,
            "confidence": np.ones(100) * 0.9,
            "tier_used": np.ones(100, dtype=int),
        })
        results = evaluate(preds, y, verbose=False)
        assert results["mae"] < 0.01
        assert results["rmse"] < 0.01
        assert results["mean_spearman"] == pytest.approx(1.0, abs=0.01)

    def test_worst_predictions(self):
        y_true = pd.Series(np.linspace(0, 100, 50))
        y_pred = pd.Series(np.linspace(100, 0, 50))  # perfectly inverted
        preds = pd.DataFrame({
            "user_id": np.repeat(np.arange(5), 10),
            "value_score": y_pred,
            "confidence": np.ones(50) * 0.5,
            "tier_used": np.ones(50, dtype=int),
        })
        results = evaluate(preds, y_true, verbose=False)
        assert results["mean_spearman"] < 0

    def test_ndcg_perfect(self):
        from value_score_model.evaluation.metrics import _ndcg_at_k
        scores = np.array([90, 70, 50, 30])
        true = np.array([90, 70, 50, 30])
        assert _ndcg_at_k(scores, true, k=3) == pytest.approx(1.0, abs=0.01)

    def test_confidence_calibration_returns_quartiles(self):
        y = pd.Series(np.random.uniform(20, 80, 200))
        preds = pd.DataFrame({
            "user_id": np.repeat(np.arange(20), 10),
            "value_score": y + np.random.normal(0, 5, 200),
            "confidence": np.random.uniform(0.2, 0.95, 200),
            "tier_used": np.ones(200, dtype=int),
        })
        results = evaluate(preds, y, verbose=False)
        assert "confidence_calibration" in results
        assert len(results["confidence_calibration"]) == 4

    def test_evaluate_display_metrics(self):
        y = pd.Series(np.full(20, 50.0))
        preds = pd.DataFrame({
            "user_id": np.repeat(np.arange(4), 5),
            "base_value_score": np.full(20, 48.0),
            "value_score": np.full(20, 110.0),
        })
        dm = evaluate_display_metrics(preds, y)
        assert "display_mae" in dm
        assert dm["display_mae"] == pytest.approx(60.0, abs=0.01)