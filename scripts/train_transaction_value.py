#!/usr/bin/env python3
"""
Train TransactionValueModel (per-transaction).

- With ``--data-dir``: loads train.py-style CSVs. If ``transaction_valuations.csv``
  joins to enough rows (see ``transaction_training.min_supervised_rows``), trains
  on supervised ``value_score``; otherwise uses the handcrafted proxy target.
- With ``--synthetic``: uses the sample or harness generator (proxy target only).

Train/test split is **grouped by user_id** to reduce user leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

from datetime import datetime

from value_score_model.config import load_config
from value_score_model.data.sample_generator import (
    generate_eval_harness_dataset,
    generate_sample_dataset,
)
from value_score_model.models.transaction_value_model import TransactionValueModel
from value_score_model.pipeline.txn_feature_engineering import (
    build_transaction_supervised_frame,
    build_transaction_training_frame,
)


def _load_training_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    keys = [
        "merchants",
        "subscriptions",
        "transactions",
        "user_explicit",
        "user_computed",
        "user_inferred",
        "user_facts",
        "feedback_signals",
        "transaction_valuations",
    ]
    data: dict[str, pd.DataFrame] = {}
    for key in keys:
        path = data_dir / f"{key}.csv"
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            if key != "transaction_valuations":
                print(f"  Warning: {key}.csv not found, skipping")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transaction-level value model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with merchants.csv, transactions.csv, user_*.csv, "
        "optional transaction_valuations.csv",
    )
    parser.add_argument(
        "--synthetic",
        choices=["sample", "harness"],
        default="harness",
        help="Synthetic dataset (ignored if --data-dir is set)",
    )
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/transaction_value.pkl"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--force-proxy",
        action="store_true",
        help="Always use handcrafted proxy y even if supervised labels exist",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ref = datetime(2024, 6, 15, 12, 0, 0)
    tt_cfg = config.get("transaction_training", {})
    min_sup = int(tt_cfg.get("min_supervised_rows", 50))

    if args.data_dir is not None:
        print(f"Loading CSVs from {args.data_dir}...")
        data = _load_training_csvs(args.data_dir)
        use_supervised = False
        if not args.force_proxy:
            sup = build_transaction_supervised_frame(data, config)
            if sup is not None:
                X, y, groups = sup
                if len(X) >= min_sup:
                    use_supervised = True
                else:
                    print(
                        f"  Supervised rows {len(X)} < min_supervised_rows {min_sup}; "
                        "using proxy target."
                    )
        if not use_supervised:
            X, y, groups = build_transaction_training_frame(data, reference_time=ref)
            label_note = "proxy y"
        else:
            label_note = "supervised value_score (mapped to learned scale)"
    else:
        if args.synthetic == "sample":
            data = generate_sample_dataset(
                n_users=args.n_users, n_merchants=20, seed=args.seed
            )
        else:
            data, _profile = generate_eval_harness_dataset(
                n_users=args.n_users, seed=args.seed, reference_time=ref
            )
        X, y, groups = build_transaction_training_frame(data, reference_time=ref)
        label_note = "proxy y"

    if len(X) < 20:
        raise SystemExit("Not enough transactions for training")

    gss = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if len(X_train) < 10:
        raise SystemExit("Not enough training rows after group split")

    model = TransactionValueModel(config)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"TransactionValueModel MAE ({label_note}): {mae:.3f}")
    print(
        f"  n_train={len(X_train)} n_test={len(X_test)} "
        f"features={list(X.columns)}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
