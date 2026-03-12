
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit

from value_score_model.models.value_score_model import ValueScoreModel
from value_score_model.pipeline.feature_engineering import FeatureEngineer, build_target
from value_score_model.data.sample_generator import generate_sample_dataset
from value_score_model.evaluation.metrics import evaluate
from value_score_model.config import load_config


def split_data(
    data: dict[str, pd.DataFrame],
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[dict, dict, dict]:

    subs = data["subscriptions"].copy()
    user_ids = subs["user_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(user_ids)

    n = len(user_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_users = set(user_ids[:n_test])
    val_users = set(user_ids[n_test:n_test + n_val])
    train_users = set(user_ids[n_test + n_val:])

    def subset(users: set) -> dict[str, pd.DataFrame]:
        d = {}
        for key, df in data.items():
            if "user_id" in df.columns:
                d[key] = df[df["user_id"].isin(users)].reset_index(drop=True)
            else:
                d[key] = df  
        return d

    return subset(train_users), subset(val_users), subset(test_users)


def main():
    parser = argparse.ArgumentParser(description="Train ValueScoreModel")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic generated data")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--n-merchants", type=int, default=20)
    parser.add_argument("--data-dir", default=None, help="Path to CSVs if not synthetic")
    parser.add_argument("--model-dir", default="checkpoints/", help="Where to save model")
    parser.add_argument("--config", default=None, help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"]["random_state"] = args.seed


    if args.synthetic:
        print(f"\nGenerating synthetic dataset: {args.n_users} users, {args.n_merchants} merchants...")
        data = generate_sample_dataset(
            n_users=args.n_users,
            n_merchants=args.n_merchants,
            seed=args.seed,
        )
        for key, df in data.items():
            print(f"  {key:25s}: {len(df):5d} rows")
    else:
        if not args.data_dir:
            raise ValueError("Provide --data-dir or --synthetic")
        data = _load_from_dir(Path(args.data_dir))


    print("\nSplitting into train / val / test (user-level)...")
    train_data, val_data, test_data = split_data(
        data,
        val_frac=config["training"]["val_split"],
        test_frac=config["training"]["test_split"],
        seed=args.seed,
    )
    print(f"  Train users: {train_data['subscriptions']['user_id'].nunique()}")
    print(f"  Val users:   {val_data['subscriptions']['user_id'].nunique()}")
    print(f"  Test users:  {test_data['subscriptions']['user_id'].nunique()}")


    model = ValueScoreModel(config)
    model.fit(train_data, val_data=val_data)


    print("Evaluating on test set...")
    fe = model.feature_engineer

    X_test, meta_test = fe.transform(test_data)
    base_test = model._rebuild_base_for_target(test_data, meta_test)
    y_test = build_target(base_test, test_data["user_computed"], config)

    predictions = model.predict(test_data)
    predictions_aligned = predictions.reset_index(drop=True)
    y_test_aligned = y_test.reset_index(drop=True).iloc[:len(predictions_aligned)]

    metrics = evaluate(predictions_aligned, y_test_aligned, verbose=True)


    model_dir = Path(args.model_dir)
    model.save(model_dir)

    metrics_path = model_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:

        safe_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                safe_metrics[k] = {str(kk): float(vv) for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                safe_metrics[k] = float(v)
            else:
                safe_metrics[k] = v
        json.dump(safe_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def _load_from_dir(data_dir: Path) -> dict[str, pd.DataFrame]:
    keys = [
        "merchants", "subscriptions", "transactions",
        "user_explicit", "user_computed", "user_inferred",
        "user_facts", "feedback_signals"
    ]
    data = {}
    for key in keys:
        path = data_dir / f"{key}.csv"
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            print(f"  Warning: {key}.csv not found, skipping")
    return data


if __name__ == "__main__":
    main()