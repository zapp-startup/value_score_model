from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Union
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .subscription_preprocessor import apply_subscription_preprocessor




MERCHANT_CATEGORY_VOCAB = [
    "streaming", "grocery", "fitness", "software",
    "utilities", "food", "education", "other"
]
BILLING_CYCLE_VOCAB = ["weekly", "monthly", "yearly", "other"]
FINANCIAL_GOAL_VOCAB = ["save_more", "invest", "reduce_debt", "build_credit", "control_subs", "other"]
RISK_TOLERANCE_VOCAB = ["low", "medium", "high"]
BUDGET_STYLE_VOCAB = ["strict", "flexible", "optimize_value"]
LIFE_STAGE_VOCAB = ["student", "early_career", "mid_career", "family", "pre_retirement", "retirement", "other"]
SUB_STATUS_VOCAB = ["active", "paused", "canceled"]


def _as_utc_timestamp(ref: Union[datetime, pd.Timestamp]) -> pd.Timestamp:
    """Reference instant in UTC for comparisons with tz-aware DB timestamps."""
    ts = pd.Timestamp(ref)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _encode_categorical(series: pd.Series, vocab: list[str]) -> pd.Series:
    """Map string category to integer index. Unknown → 0."""
    mapping = {v: i + 1 for i, v in enumerate(vocab)}
    return series.map(mapping).fillna(0).astype(int)


def _safe_fill(df: pd.DataFrame, col: str, fill_value: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(fill_value, index=df.index)
    return df[col].fillna(fill_value)


class DecayComputer:
    """Stateless utility: computes usage and time decay multipliers."""

    def __init__(self, usage_lambda: float = 0.015, time_lambda: float = 0.003):
        self.usage_lambda = usage_lambda
        self.time_lambda = time_lambda

    def usage_decay(self, days_since_last_used: pd.Series) -> pd.Series:
        """Primary decay: driven by how long since last use."""
        return np.exp(-self.usage_lambda * days_since_last_used.clip(lower=0))

    def time_decay(self, days_since_signal: pd.Series) -> pd.Series:
        """Secondary mild decay: newer signals matter slightly more."""
        return np.exp(-self.time_lambda * days_since_signal.clip(lower=0))


class FeatureEngineer:


    NUMERICAL_FEATURES = [
        "price_norm",
        "days_since_started",
        "usage_frequency",
        "days_since_last_used",
        "reactivation_count",
        "transaction_count_30d",
        "avg_amount_30d",
        "impulse_score",
        "regret_score",
        # User-level
        "cost_weight",
        "quality_weight",
        "sustainability_weight",
        "impulse_susceptibility_score",
        "regret_sensitivity",
        "budget_adherence_score",
        "percent_income_on_subs",
        "regret_frequency",
        "active_subscriptions_count_norm",
        # Feedback
        "satisfaction_rating_norm",
        "regret_rating_norm_inv",   
        "repurchase_likelihood_norm",
        "text_feedback_score",
        "text_feedback_confidence",
        # Decay multipliers
        "usage_decay",
        "time_decay",
        # Generic subscription signals (from subscription_preprocessor; raw metrics vary by merchant)
        "subscription_utilization",
        "subscription_cost_benefit",
    ]

    CATEGORICAL_EMBEDDING_FEATURES = [
        "merchant_category_id",
        "billing_cycle_id",
        "financial_goal_id",
        "risk_tolerance_id",
        "budget_style_id",
        "life_stage_id",
        "sub_status_id",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.decay = DecayComputer(
            usage_lambda=config["decay"]["usage_lambda"],
            time_lambda=config["decay"]["time_lambda"],
        )
        self.scaler = StandardScaler()
        self._fitted = False



    def fit(self, data: dict[str, pd.DataFrame], reference_time: Optional[datetime] = None) -> "FeatureEngineer":
        """Fit scaler on training data. Call once before transform."""
        raw, _ = self._build_raw_features(
            data, reference_time or datetime.now(timezone.utc)
        )
        num_cols = [c for c in self.NUMERICAL_FEATURES if c in raw.columns]
        self.scaler.fit(raw[num_cols].fillna(0).values)
        self._fitted = True
        self._num_cols_order = num_cols
        return self

    def transform(
        self,
        data: dict[str, pd.DataFrame],
        reference_time: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        if not self._fitted:
            raise RuntimeError("Call fe.fit() before fe.transform()")

        ref = reference_time or datetime.now(timezone.utc)
        raw, meta = self._build_raw_features(data, ref)


        num_cols = [c for c in self._num_cols_order if c in raw.columns]
        scaled = self.scaler.transform(raw[num_cols].fillna(0).values)
        scaled_df = pd.DataFrame(scaled, columns=num_cols, index=raw.index)


        cat_cols = [c for c in self.CATEGORICAL_EMBEDDING_FEATURES if c in raw.columns]
        result = pd.concat([scaled_df, raw[cat_cols]], axis=1)

        return result, meta

    def fit_transform(
        self,
        data: dict[str, pd.DataFrame],
        reference_time: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(data, reference_time)
        return self.transform(data, reference_time)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEngineer":
        return joblib.load(path)



    def _build_raw_features(
        self,
        data: dict[str, pd.DataFrame],
        ref: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        data = apply_subscription_preprocessor(data, self.config)
        subs = data["subscriptions"].copy()
        txns = data["transactions"].copy()
        merchants = data["merchants"].copy()
        user_exp = data["user_explicit"].copy()
        user_comp = data["user_computed"].copy()
        user_inf = data["user_inferred"].copy()
        user_facts = data.get("user_facts", pd.DataFrame())
        feedback = data.get("feedback_signals", pd.DataFrame())


        if "direction" in txns.columns:
            txns = txns[txns["direction"] == "spend"].copy()

        ref_ts = _as_utc_timestamp(ref)
        txns["occurred_at"] = pd.to_datetime(txns["occurred_at"], utc=True)

        cutoff_30d = ref_ts - pd.Timedelta(days=30)
        txns_30d = txns[txns["occurred_at"] >= cutoff_30d]

        agg_30d = txns_30d.groupby(["user_id", "merchant_id"]).agg(
            transaction_count_30d=("id", "count"),
            avg_amount_30d=("amount", "mean"),
        ).reset_index()


        latest_txn = txns.groupby(["user_id", "merchant_id"])["occurred_at"].max().reset_index()
        latest_txn.columns = ["user_id", "merchant_id", "last_used_at"]


        if not txns.empty and "satisfaction_rating" in txns.columns:
            fb_agg = txns.groupby(["user_id", "merchant_id"]).agg(
                satisfaction_rating=("satisfaction_rating", "mean"),
                regret_rating=("regret_rating", "mean"),
                repurchase_likelihood=("repurchase_likelihood", "mean"),
                impulse_score=("impulse_score", "mean"),
                regret_score=("regret_score", "mean"),
                txn_usage_frequency=("usage_frequency", "mean"),
            ).reset_index()
        else:
            fb_agg = subs[["user_id", "merchant_id"]].drop_duplicates()
            for c in ["satisfaction_rating", "regret_rating", "repurchase_likelihood",
                      "impulse_score", "regret_score", "txn_usage_frequency"]:
                fb_agg[c] = np.nan


        base = subs.merge(merchants[["id", "category"]], left_on="merchant_id", right_on="id", suffixes=("", "_m"))
        base = base.rename(columns={"category": "merchant_category"})


        base["started_on"] = pd.to_datetime(base.get("started_on", pd.NaT), utc=True)
        base["days_since_started"] = (
            (ref_ts - base["started_on"]).dt.days.fillna(180).clip(lower=0)
        )


        base = base.merge(agg_30d, on=["user_id", "merchant_id"], how="left")
        base = base.merge(latest_txn, on=["user_id", "merchant_id"], how="left")
        base = base.merge(fb_agg, on=["user_id", "merchant_id"], how="left")


        base["last_used_at"] = pd.to_datetime(base.get("last_used_at", pd.NaT), utc=True)
        base["days_since_last_used"] = (
            (ref_ts - base["last_used_at"]).dt.days.fillna(999).clip(lower=0)
        )


        base["usage_frequency"] = base["txn_usage_frequency"].fillna(
            base.get("usage_frequency_x", base.get("usage_frequency", 1.0))
        ).fillna(1.0)


        base["usage_decay"] = self.decay.usage_decay(base["days_since_last_used"])
        base["time_decay"] = self.decay.time_decay(base["days_since_started"])


        base = base.merge(user_exp, on="user_id", how="left")
        base = base.merge(user_comp, on="user_id", how="left", suffixes=("", "_comp"))
        base = base.merge(user_inf, on="user_id", how="left")


        min_conf = self.config["feedback"]["userfact_min_confidence"]
        if not user_facts.empty and "confidence" in user_facts.columns:
            hc_facts = user_facts[user_facts["confidence"] >= min_conf].copy()
            if not hc_facts.empty:
                facts_pivot = hc_facts.pivot_table(
                    index="user_id", columns="fact_key",
                    values="fact_value", aggfunc="mean"
                )
                facts_pivot.columns = [f"fact_{c}" for c in facts_pivot.columns]
                base = base.merge(facts_pivot.reset_index(), on="user_id", how="left")


        min_fb_conf = self.config["feedback"]["min_confidence"]
        if not feedback.empty:
            fb = feedback[["user_id", "merchant_id", "feedback_value_score", "feedback_confidence"]].copy()
            # Subscriptions often already carry these columns; merge would create _x/_y and break lookups.
            for c in ("feedback_value_score", "feedback_confidence"):
                if c in base.columns:
                    base = base.drop(columns=[c])
            base = base.merge(fb, on=["user_id", "merchant_id"], how="left")

            base.loc[
                base["feedback_confidence"].fillna(0) < min_fb_conf,
                ["feedback_value_score", "feedback_confidence"]
            ] = 0.0
        else:
            if "feedback_value_score" not in base.columns:
                base["feedback_value_score"] = 0.0
            if "feedback_confidence" not in base.columns:
                base["feedback_confidence"] = 0.0

        base["text_feedback_score"] = _safe_fill(base, "feedback_value_score", 0.0)
        base["text_feedback_confidence"] = _safe_fill(base, "feedback_confidence", 0.0)


        monthly_income = base["monthly_income"].replace(0, np.nan).fillna(3000)
        base["price_norm"] = (base["price"] / monthly_income).clip(0, 1)

        base["satisfaction_rating_norm"] = (
            _safe_fill(base, "satisfaction_rating", 5.0) / 10.0
        )
        base["regret_rating_norm_inv"] = 1.0 - (
            _safe_fill(base, "regret_rating", 50.0) / 100.0
        )
        base["repurchase_likelihood_norm"] = (
            _safe_fill(base, "repurchase_likelihood", 50.0) / 100.0
        )
        base["active_subscriptions_count_norm"] = (
            _safe_fill(base, "active_subscriptions_count", 3.0).clip(0, 20) / 20.0
        )
        base["transaction_count_30d"] = _safe_fill(base, "transaction_count_30d", 0)
        base["avg_amount_30d"] = _safe_fill(base, "avg_amount_30d", base.get("price", 0))
        base["impulse_score"] = _safe_fill(base, "impulse_score", 0.0)
        base["regret_score"] = _safe_fill(base, "regret_score", 0.0)
        base["reactivation_count"] = _safe_fill(base, "reactivation_count", 0)
        base["percent_income_on_subs"] = _safe_fill(
            base, "percent_income_spent_on_subscriptions", 0.05
        )

        base["subscription_utilization"] = _safe_fill(
            base, "subscription_utilization",
            float(self.config.get("subscription_signals", {}).get("default_utilization", 0.5)),
        ).clip(0, 1)
        base["subscription_cost_benefit"] = _safe_fill(
            base, "subscription_cost_benefit",
            float(self.config.get("subscription_signals", {}).get("default_cost_benefit", 0.0)),
        ).clip(0, 1)

        base["merchant_category_id"] = _encode_categorical(
            base.get("merchant_category", pd.Series(["other"] * len(base))),
            MERCHANT_CATEGORY_VOCAB
        )
        base["billing_cycle_id"] = _encode_categorical(
            _safe_col(base, "billing_cycle", "monthly"), BILLING_CYCLE_VOCAB
        )
        base["financial_goal_id"] = _encode_categorical(
            _safe_col(base, "financial_goal", "other"), FINANCIAL_GOAL_VOCAB
        )
        base["risk_tolerance_id"] = _encode_categorical(
            _safe_col(base, "risk_tolerance", "medium"), RISK_TOLERANCE_VOCAB
        )
        base["budget_style_id"] = _encode_categorical(
            _safe_col(base, "budget_style", "flexible"), BUDGET_STYLE_VOCAB
        )
        base["life_stage_id"] = _encode_categorical(
            _safe_col(base, "life_stage", "other"), LIFE_STAGE_VOCAB
        )
        base["sub_status_id"] = _encode_categorical(
            _safe_col(base, "status", "active"), SUB_STATUS_VOCAB
        )


        total_txns = txns.groupby(["user_id", "merchant_id"])["id"].count().reset_index()
        total_txns.columns = ["user_id", "merchant_id", "total_transaction_count"]
        base = base.merge(total_txns, on=["user_id", "merchant_id"], how="left")
        base["total_transaction_count"] = base["total_transaction_count"].fillna(0)

        meta = base[
            [
                "user_id",
                "merchant_id",
                "id",
                "total_transaction_count",
                "transaction_count_30d",
                "usage_frequency",
                "days_since_last_used",
                "days_since_started",
                "billing_cycle_id",
                "usage_decay",
            ]
        ].copy()
        meta = meta.rename(columns={"id": "subscription_id"})

        return base, meta


def _safe_col(df: pd.DataFrame, col: str, default: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def build_target(
    base_df: pd.DataFrame,
    user_computed: pd.DataFrame,
    config: dict,
) -> pd.Series:
    
    if "cost_weight" not in base_df.columns:
        df = base_df.merge(
            user_computed[["user_id", "cost_weight", "quality_weight", "sustainability_weight"]],
            on="user_id", how="left"
        )
    else:
        df = base_df.copy()

    df = df.reset_index(drop=True)
    cost_w = df["cost_weight"].fillna(0.33)
    qual_w = df["quality_weight"].fillna(0.33)
    sust_w = df["sustainability_weight"].fillna(0.34)


    if "price_norm" in df.columns:
        price_value = 1.0 - df["price_norm"].clip(0, 1)
    elif "price" in df.columns:

        price = df["price"].fillna(20.0)
        income = df["monthly_income"].fillna(3000.0) if "monthly_income" in df.columns else 3000.0
        price_value = 1.0 - (price / income).clip(0, 1)
    else:
        price_value = pd.Series(0.5, index=df.index)


    sat = df["satisfaction_rating"].fillna(5.0) / 10.0 if "satisfaction_rating" in df.columns \
        else pd.Series(0.5, index=df.index)
    repurchase = df["repurchase_likelihood"].fillna(50.0) / 100.0 if "repurchase_likelihood" in df.columns \
        else pd.Series(0.5, index=df.index)
    regret_inv = 1.0 - (df["regret_rating"].fillna(50.0) / 100.0) if "regret_rating" in df.columns \
        else pd.Series(0.5, index=df.index)


    sustainability_signal = (repurchase + regret_inv) / 2.0

    target_01 = (
        cost_w * price_value +
        qual_w * sat +
        sust_w * sustainability_signal
    ).clip(0, 1)


    if "usage_decay" in df.columns:
        usage_decay = df["usage_decay"].clip(0, 1).fillna(1.0)
        target_01 = (target_01 * usage_decay).clip(0, 1)

    return (target_01 * 100).round(1)