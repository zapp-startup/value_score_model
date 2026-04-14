#scoring model picks between the three

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import joblib

from ..pipeline.feature_engineering import FeatureEngineer, build_target
from ..pipeline.overutilisation import (
    apply_overutilisation_guardrails,
    compute_overutilisation_bonus,
)
from ..config import load_config, scoring_scale
from .tier1_coldstart import ColdStartModel
from .tier2_xgboost import XGBoostValueModel, BLEND_WINDOW
from .tier3_neural import NeuralValueModel


class ValueScoreModel:

    TIER1_MAX = None  
    TIER2_MAX = None  

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.TIER1_MAX = self.config["tier"]["cold_start_max_transactions"]
        self.TIER2_MAX = self.config["tier"]["baseline_max_transactions"]

        self.feature_engineer = FeatureEngineer(self.config)
        self.tier1 = ColdStartModel(self.config)
        self.tier2 = XGBoostValueModel(self.config)
        self.tier3 = NeuralValueModel(self.config)

        self._fitted = False



    def fit(
        self,
        data: dict[str, pd.DataFrame],
        reference_time=None,
        val_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> "ValueScoreModel":
        """
        Full training pass. Fits feature engineer, then all three tiers.

        data:      dict with keys: merchants, subscriptions, transactions,
                   user_explicit, user_computed, user_inferred,
                   user_facts (optional), feedback_signals (optional)
        val_data:  optional held-out set with same structure
        """
        print("=== ValueScoreModel: fitting ===")


        print("[1/4] Feature engineering...")
        X, meta = self.feature_engineer.fit_transform(data, reference_time)
        base_df = self._rebuild_base_for_target(data, meta, reference_time)
        y = build_target(base_df, data["user_computed"], self.config)
        y = y.reset_index(drop=True).iloc[:len(X)]

        print(f"      Built {len(X)} training samples, {X.shape[1]} features")
        print(f"      Target range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")


        X_val, meta_val, y_val = None, None, None
        if val_data:
            X_val, meta_val = self.feature_engineer.transform(val_data, reference_time)
            base_val = self._rebuild_base_for_target(val_data, meta_val, reference_time)
            y_val = build_target(base_val, val_data["user_computed"], self.config)


        print("[2/4] Fitting Tier 1 (cold-start population model)...")
        base_df_full = self._get_base_df(data, meta, X)
        self.tier1.fit(base_df_full, y)


        print("[3/4] Fitting Tier 2 (XGBoost)...")
        eval_set = None
        if X_val is not None:
            eval_set = (X_val, y_val)
        self.tier2.fit(X, y, eval_set=eval_set)


        tier3_mask = meta["total_transaction_count"] >= self.TIER2_MAX
        if tier3_mask.sum() >= 20:
            print(f"[4/4] Fitting Tier 3 (neural) on {tier3_mask.sum()} high-data samples...")
            user_ids = meta["user_id"].values
            merchant_ids = meta["merchant_id"].values

            val_kwargs = {}
            if X_val is not None:
                val_kwargs = dict(
                    X_val=X_val, y_val=y_val,
                    user_ids_val=meta_val["user_id"].values,
                    merchant_ids_val=meta_val["merchant_id"].values,
                )
            self.tier3.fit(
                X, y,
                user_ids=user_ids,
                merchant_ids=merchant_ids,
                **val_kwargs,
            )
        else:
            print("[4/4] Tier 3 skipped — insufficient high-data samples (<20). Will use Tier 2 for all.")

        self._fitted = True
        print("=== Fitting complete ===\n")
        return self



    def predict(
        self,
        data: dict[str, pd.DataFrame],
        reference_time=None,
    ) -> pd.DataFrame:
        
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X, meta = self.feature_engineer.transform(data, reference_time)
        txn_counts = meta["total_transaction_count"].values
        user_ids = meta["user_id"].values
        merchant_ids = meta["merchant_id"].values

        base_df = self._get_base_df(data, meta, X)

        n = len(X)
        base_scores = np.zeros(n, dtype=np.float32)
        final_confidences = np.zeros(n, dtype=np.float32)
        tier_used = np.zeros(n, dtype=np.int8)


        t1_mask = txn_counts < self.TIER1_MAX
        t2_mask = (txn_counts >= self.TIER1_MAX) & (txn_counts < self.TIER2_MAX)
        t3_mask = txn_counts >= self.TIER2_MAX


        if t1_mask.any():
            s1, c1 = self.tier1.predict_with_confidence(base_df[t1_mask])
            base_scores[t1_mask] = s1
            final_confidences[t1_mask] = c1
            tier_used[t1_mask] = 1


        if t2_mask.any():
            s2, c2 = self.tier2.predict_with_confidence(X[t2_mask], txn_counts[t2_mask])
            s1_blend, _ = self.tier1.predict_with_confidence(base_df[t2_mask])
            alpha = np.clip(txn_counts[t2_mask] / BLEND_WINDOW, 0, 1)
            blended = alpha * s2 + (1 - alpha) * s1_blend
            base_scores[t2_mask] = blended
            final_confidences[t2_mask] = c2
            tier_used[t2_mask] = 2


        if t3_mask.any():
            try:
                s3, c3 = self.tier3.predict_with_confidence(
                    X[t3_mask], user_ids[t3_mask], merchant_ids[t3_mask],
                    txn_counts=txn_counts[t3_mask],
                )
                base_scores[t3_mask] = s3
                final_confidences[t3_mask] = c3
                tier_used[t3_mask] = 3
            except RuntimeError:

                s2_fb, c2_fb = self.tier2.predict_with_confidence(
                    X[t3_mask], txn_counts[t3_mask]
                )
                base_scores[t3_mask] = s2_fb
                final_confidences[t3_mask] = c2_fb
                tier_used[t3_mask] = 2

        scales = scoring_scale(self.config)
        learned_max = scales["learned_max"]
        display_max = scales["display_max"]
        base_scores = np.clip(base_scores, 0, learned_max)
        over_bonus = compute_overutilisation_bonus(meta, self.config)
        over_bonus = apply_overutilisation_guardrails(base_scores, over_bonus, self.config)
        final_scores = np.clip(base_scores + over_bonus, 0, display_max)

        results = pd.DataFrame({
            "user_id": user_ids,
            "merchant_id": merchant_ids,
            "subscription_id": meta["subscription_id"].values,
            "base_value_score": base_scores.round().astype(int),
            "overutilisation_bonus": over_bonus.round(1),
            "value_score": final_scores.round().astype(int),
            "display_value_score": final_scores.round().astype(int),
            "confidence": final_confidences.round(3),
            "tier_used": tier_used,
        })

        if self.config.get("scoring", {}).get("save_evidence", True):
            results["evidence_json"] = self._build_evidence(
                X, meta, base_scores, over_bonus, final_scores, tier_used
            )

        return results



    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.feature_engineer.save(d / "feature_engineer.pkl")
        joblib.dump(self.tier1, d / "tier1_coldstart.pkl")
        self.tier2.save(d / "tier2_xgboost.pkl")
        if self.tier3._fitted:
            self.tier3.save(d / "tier3_neural.pt")
        joblib.dump({"config": self.config, "fitted": self._fitted}, d / "meta.pkl")
        print(f"Model saved to {d}/")

    @classmethod
    def load(cls, directory: str | Path, config: Optional[dict] = None) -> "ValueScoreModel":
        d = Path(directory)
        meta = joblib.load(d / "meta.pkl")
        cfg = config or meta["config"]
        m = cls(cfg)
        m.feature_engineer = FeatureEngineer.load(d / "feature_engineer.pkl")
        m.tier1 = joblib.load(d / "tier1_coldstart.pkl")
        m.tier2 = XGBoostValueModel.load(d / "tier2_xgboost.pkl")
        if (d / "tier3_neural.pt").exists():
            m.tier3 = NeuralValueModel.load(d / "tier3_neural.pt", cfg)
        m._fitted = meta["fitted"]
        print(f"Model loaded from {d}/")
        return m



    def _rebuild_base_for_target(
        self,
        data: dict,
        meta: pd.DataFrame,
        reference_time=None,
    ) -> pd.DataFrame:
        """Get raw base dataframe with un-scaled features for target building."""
        subs = data["subscriptions"].copy()
        txns = data["transactions"].copy()
        if "direction" in txns.columns:
            txns = txns[txns["direction"] == "spend"]


        if "satisfaction_rating" in txns.columns:
            agg = txns.groupby(["user_id", "merchant_id"]).agg(
                satisfaction_rating=("satisfaction_rating", "mean"),
                regret_rating=("regret_rating", "mean"),
                repurchase_likelihood=("repurchase_likelihood", "mean"),
            ).reset_index()
            base = subs.merge(agg, on=["user_id", "merchant_id"], how="left")
        else:
            base = subs.copy()

        base = base.merge(
            data["user_explicit"][["user_id", "monthly_income"]],
            on="user_id", how="left"
        )
        monthly_income = base["monthly_income"].replace(0, None).fillna(3000)
        base["price_norm"] = (base["price"] / monthly_income).clip(0, 1)

        from ..pipeline.feature_engineering import DecayComputer, _as_utc_timestamp
        import datetime as _dt

        dc = DecayComputer(
            self.config["decay"]["usage_lambda"],
            self.config["decay"]["time_lambda"],
        )
        # Align with feature_engineering / fit(reference_time); wall clock would
        # make historical txns look "stale" and zero out usage_decay in targets.
        ref_ts = (
            _as_utc_timestamp(reference_time)
            if reference_time is not None
            else _as_utc_timestamp(_dt.datetime.now(_dt.timezone.utc))
        )
        txns["occurred_at"] = pd.to_datetime(
            txns["occurred_at"], utc=True, format="mixed"
        )
        latest = txns.groupby(["user_id", "merchant_id"])["occurred_at"].max().reset_index()
        latest.columns = ["user_id", "merchant_id", "last_used_at"]
        base = base.merge(latest, on=["user_id", "merchant_id"], how="left")
        base["last_used_at"] = pd.to_datetime(base["last_used_at"], utc=True)
        base["days_since_last_used"] = (
            (ref_ts - base["last_used_at"]).dt.days.fillna(999).clip(lower=0)
        )
        base["usage_decay"] = dc.usage_decay(base["days_since_last_used"])

        return base

    def _get_base_df(
        self, data: dict, meta: pd.DataFrame, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Reconstruct a base df aligned with X for cold-start tier."""
        subs = data["subscriptions"].copy()
        merchants = data["merchants"][["id", "category"]].rename(
            columns={"id": "merchant_id", "category": "merchant_category"}
        )
        base = subs.merge(merchants, on="merchant_id", how="left")
        base = base.merge(
            data["user_explicit"][[
                "user_id", "life_stage", "financial_goal",
                "monthly_income", "value_priority_cost",
                "value_priority_quality", "value_priority_sustainability"
            ]],
            on="user_id", how="left"
        )
        base = base.merge(
            data["user_computed"][["user_id", "cost_weight", "quality_weight"]],
            on="user_id", how="left"
        )
        monthly_income = base["monthly_income"].replace(0, None).fillna(3000)
        base["price_norm"] = (base["price"] / monthly_income).clip(0, 1)
        base = base.merge(meta[["user_id", "merchant_id", "subscription_id"]], 
                          left_on=["user_id", "merchant_id", "id"],
                          right_on=["user_id", "merchant_id", "subscription_id"],
                          how="inner")
        base = base.reset_index(drop=True)
        return base

    def _build_evidence(
        self,
        X: pd.DataFrame,
        meta: pd.DataFrame,
        base_scores: np.ndarray,
        over_bonus: np.ndarray,
        final_scores: np.ndarray,
        tiers: np.ndarray,
    ) -> list[dict]:
        key_features = [
            "cost_weight", "quality_weight", "sustainability_weight",
            "satisfaction_rating_norm", "regret_rating_norm_inv",
            "repurchase_likelihood_norm", "usage_decay", "time_decay",
            "text_feedback_score", "text_feedback_confidence",
            "transaction_count_30d",
            "subscription_utilization", "subscription_cost_benefit",
        ]
        evidence = []
        for i, row in X.iterrows():
            snap = {col: round(float(row[col]), 4) for col in key_features if col in X.columns}
            snap["tier_used"] = int(tiers[i] if i < len(tiers) else 0)
            snap["base_value_score"] = round(float(base_scores[i] if i < len(base_scores) else 0), 2)
            snap["overutilisation_bonus"] = round(float(over_bonus[i] if i < len(over_bonus) else 0), 2)
            snap["final_score"] = round(float(final_scores[i] if i < len(final_scores) else 0), 2)
            evidence.append(snap)
        return evidence