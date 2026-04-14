"""
Join helpers and integrity checks for exported valuation tables vs core entities.

Primary use: ``valuations_transactionvaluation`` rows keyed by ``transaction_id``
must align with ``transactions_transaction.id`` after ``build_training_data.py``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Minimum columns to treat an export as transaction-level supervision.
REQUIRED_TRANSACTION_VALUATION_COLUMNS = (
    "transaction_id",
    "user_id",
    "value_score",
)

REQUIRED_SUBSCRIPTION_VALUATION_COLUMNS = (
    "subscription_id",
    "user_id",
)

REQUIRED_ITEM_VALUATION_COLUMNS = (
    "id",
    "user_id",
    "personal_value_score",
)


def assert_transaction_valuation_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_TRANSACTION_VALUATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "transaction valuations DataFrame missing required columns: "
            f"{missing}; required={list(REQUIRED_TRANSACTION_VALUATION_COLUMNS)}"
        )


def assert_subscription_valuation_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SUBSCRIPTION_VALUATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "subscription valuations DataFrame missing required columns: "
            f"{missing}; required={list(REQUIRED_SUBSCRIPTION_VALUATION_COLUMNS)}"
        )


def assert_item_valuation_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_ITEM_VALUATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "item valuations DataFrame missing required columns: "
            f"{missing}; required={list(REQUIRED_ITEM_VALUATION_COLUMNS)}"
        )


def _numeric_ids(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna().astype("int64")


def transaction_valuation_join_stats(
    transactions: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    transaction_pk_col: str = "id",
) -> dict[str, Any]:
    """
    Compare transaction PKs to ``valuations.transaction_id``.

    Returns counts for coverage reporting and tests. Does not require one-to-one
    (multiple valuation rows per transaction_id are allowed).
    """
    assert_transaction_valuation_schema(valuations)
    if transaction_pk_col not in transactions.columns:
        raise ValueError(f"transactions missing column {transaction_pk_col!r}")

    tx_ids = set(_numeric_ids(transactions[transaction_pk_col]).unique())
    val_tx_ids = set(_numeric_ids(valuations["transaction_id"]).unique())

    matched = val_tx_ids & tx_ids
    orphan_val = val_tx_ids - tx_ids

    denom = len(val_tx_ids) if val_tx_ids else 1
    return {
        "n_distinct_transaction_pks": len(tx_ids),
        "n_distinct_valuation_transaction_ids": len(val_tx_ids),
        "n_valuation_rows": int(len(valuations)),
        "n_matched_valuation_transaction_ids": len(matched),
        "n_orphan_valuation_transaction_ids": len(orphan_val),
        "orphan_valuation_transaction_ids": sorted(orphan_val),
        "pct_valuation_txn_ids_found_in_transactions": len(matched) / denom,
    }


def orphan_transaction_valuation_ids(
    transactions: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    transaction_pk_col: str = "id",
) -> list[int]:
    stats = transaction_valuation_join_stats(
        transactions, valuations, transaction_pk_col=transaction_pk_col
    )
    return list(stats["orphan_valuation_transaction_ids"])


def merge_transaction_valuations_left(
    transactions: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    transaction_pk_col: str = "id",
    valuations_suffix: str = "_val",
) -> pd.DataFrame:
    """Left-join valuations onto transactions (one row per transaction row)."""
    assert_transaction_valuation_schema(valuations)
    v = valuations.copy()
    if valuations_suffix:
        dup_cols = [c for c in v.columns if c in transactions.columns and c != transaction_pk_col]
        rename = {c: f"{c}{valuations_suffix}" for c in dup_cols}
        v = v.rename(columns=rename)
    return transactions.merge(
        v,
        how="left",
        left_on=transaction_pk_col,
        right_on="transaction_id",
    )
