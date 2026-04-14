"""
Per-transaction feature rows for TransactionValueModel.

- Proxy labels: handcrafted y in [0, 100] (build_target-like, no usage_decay).
- Supervised labels: join ``transaction_valuations`` on ``transactions.id`` ==
  ``transaction_id``; map ``value_score`` from [0, label_max] to learned scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime


def _encode_merchant_category_series(series: pd.Series) -> pd.Series:
    """Local helper; map category strings to int ids (same vocab as feature_engineering)."""
    from .feature_engineering import MERCHANT_CATEGORY_VOCAB

    mapping = {v: i + 1 for i, v in enumerate(MERCHANT_CATEGORY_VOCAB)}
    return series.map(mapping).fillna(0).astype(int)


def _merged_spend_transactions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spend transactions with merchant category and user preference columns."""
    tx = data["transactions"].copy()
    if "direction" in tx.columns:
        tx = tx[tx["direction"] == "spend"].copy()
    merchants = data["merchants"].rename(columns={"id": "merchant_id"})
    mcat = merchants[["merchant_id", "category"]].rename(
        columns={"category": "merchant_category"}
    )
    tx = tx.merge(mcat, on="merchant_id", how="left")
    tx["merchant_category"] = tx["merchant_category"].fillna("other")

    ue = data["user_explicit"][["user_id", "monthly_income"]].copy()
    uc = data["user_computed"][
        ["user_id", "cost_weight", "quality_weight", "sustainability_weight"]
    ].copy()
    tx = tx.merge(ue, on="user_id", how="left")
    tx = tx.merge(uc, on="user_id", how="left")
    return tx.reset_index(drop=True)


def _txn_feature_matrix(tx: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix from a prepared transaction frame (one row per txn)."""
    income = tx["monthly_income"].replace(0, np.nan).fillna(3000.0).astype(float)
    amt = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    price_norm = (amt / income).clip(0, 1)
    sat = pd.to_numeric(tx.get("satisfaction_rating"), errors="coerce").fillna(5.0) / 10.0
    rep = pd.to_numeric(tx.get("repurchase_likelihood"), errors="coerce").fillna(50.0) / 100.0
    regret = pd.to_numeric(tx.get("regret_rating"), errors="coerce").fillna(50.0)
    regret_inv = 1.0 - (regret / 100.0).clip(0, 1)

    cw = pd.to_numeric(tx["cost_weight"], errors="coerce").fillna(1 / 3)
    qw = pd.to_numeric(tx["quality_weight"], errors="coerce").fillna(1 / 3)
    sw = pd.to_numeric(tx["sustainability_weight"], errors="coerce").fillna(1 / 3)

    X = pd.DataFrame(
        {
            "amount": amt,
            "price_norm": price_norm,
            "satisfaction_norm": sat,
            "repurchase_norm": rep,
            "regret_inv_norm": regret_inv,
            "impulse_score": pd.to_numeric(tx.get("impulse_score"), errors="coerce").fillna(
                0.0
            ),
            "regret_score": pd.to_numeric(tx.get("regret_score"), errors="coerce").fillna(0.0),
            "cost_weight": cw,
            "quality_weight": qw,
            "sustainability_weight": sw,
            "merchant_category_id": _encode_merchant_category_series(
                tx["merchant_category"]
            ),
            "merchant_id": pd.to_numeric(tx["merchant_id"], errors="coerce")
            .fillna(0)
            .astype(int),
        }
    )
    return X.fillna(0).reset_index(drop=True)


def build_transaction_training_frame(
    data: dict[str, pd.DataFrame],
    reference_time: datetime | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build (X, y, user_id) for spend transactions with handcrafted proxy y in [0, 100].

    ``user_id`` aligns with each row for group-wise train/test splits.
    """
    tx = _merged_spend_transactions(data)
    income = tx["monthly_income"].replace(0, np.nan).fillna(3000.0).astype(float)
    amt = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    price_norm = (amt / income).clip(0, 1)
    price_value = 1.0 - price_norm

    sat = pd.to_numeric(tx.get("satisfaction_rating"), errors="coerce").fillna(5.0) / 10.0
    rep = pd.to_numeric(tx.get("repurchase_likelihood"), errors="coerce").fillna(50.0) / 100.0
    regret = pd.to_numeric(tx.get("regret_rating"), errors="coerce").fillna(50.0)
    regret_inv = 1.0 - (regret / 100.0).clip(0, 1)
    sustainability = (rep + regret_inv) / 2.0

    cw = pd.to_numeric(tx["cost_weight"], errors="coerce").fillna(1 / 3)
    qw = pd.to_numeric(tx["quality_weight"], errors="coerce").fillna(1 / 3)
    sw = pd.to_numeric(tx["sustainability_weight"], errors="coerce").fillna(1 / 3)

    target_01 = (cw * price_value + qw * sat + sw * sustainability).clip(0, 1)
    y = (target_01 * 100.0).round(1)

    X = _txn_feature_matrix(tx)
    user_id = pd.to_numeric(tx["user_id"], errors="coerce").fillna(0).astype(int)
    user_id = user_id.reset_index(drop=True)
    return X, y.reset_index(drop=True), user_id


def build_transaction_supervised_frame(
    data: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """
    Join ``data['transaction_valuations']`` on ``transactions.id`` == ``transaction_id``.

    Maps ``value_score`` from [0, label_max] to [0, learned_score_max] for training
    (``transaction_training.label_max`` and ``scoring.learned_score_max``).

    Returns (X, y, user_id) or None if no valuations / no overlapping rows.
    """
    vals = data.get("transaction_valuations")
    if vals is None or len(vals) == 0:
        return None
    if "transaction_id" not in vals.columns or "value_score" not in vals.columns:
        return None

    v = vals.copy()
    if "created_at" in v.columns:
        v["_ts"] = pd.to_datetime(v["created_at"], utc=True, format="mixed", errors="coerce")
        v = v.sort_values("_ts", na_position="first")
    v = v.drop_duplicates(subset=["transaction_id"], keep="last")
    v = v[["transaction_id", "value_score"]].copy()

    tx = _merged_spend_transactions(data)
    if "id" not in tx.columns:
        return None
    merged = tx.merge(v, left_on="id", right_on="transaction_id", how="inner")
    if len(merged) == 0:
        return None

    y_raw = pd.to_numeric(merged["value_score"], errors="coerce")
    ok = y_raw.notna()
    merged = merged.loc[ok].reset_index(drop=True)
    y_raw = y_raw.loc[ok].reset_index(drop=True)
    if len(merged) == 0:
        return None

    tt = config.get("transaction_training", {})
    label_max = float(tt.get("label_max", 150))
    if label_max <= 0:
        label_max = 150.0
    learned_max = float(config.get("scoring", {}).get("learned_score_max", 100))

    y = (y_raw / label_max * learned_max).clip(0, learned_max).astype(float).round(3)
    X = _txn_feature_matrix(merged)
    user_id = pd.to_numeric(merged["user_id"], errors="coerce").fillna(0).astype(int)
    user_id = user_id.reset_index(drop=True)
    return X, y, user_id
