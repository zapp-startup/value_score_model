"""
End-to-end checks for scripts/simulate_and_export_value_scores.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from datetime import datetime

from value_score_model.config import load_config

_MAX_SCORE = float(load_config().get("scoring", {}).get("max_score", 150))
from value_score_model.data.sample_generator import generate_eval_harness_dataset
from value_score_model.pipeline.feature_engineering import FeatureEngineer

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "simulate_and_export_value_scores.py"


def _run_script(n_users: int, seed: int, outdir: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--n-users",
        str(n_users),
        "--seed",
        str(seed),
        "--outdir",
        str(outdir),
    ]
    subprocess.run(cmd, cwd=str(WORKSPACE_ROOT), check=True, capture_output=True, text=True)


class TestSimulateAndExport:

    def test_outputs_created_and_counts(self, tmp_path: Path):
        n = 25
        out = tmp_path / "eval_out"
        _run_script(n, seed=11, outdir=out)

        csv_path = out / f"value_score_eval_{n}.csv"
        jsonl_path = out / f"value_score_eval_{n}.jsonl"
        summary_path = out / "value_score_eval_summary.txt"
        subs_path = out / f"value_score_eval_{n}_subscriptions.csv"
        txn_path = out / f"value_score_eval_{n}_transactions.csv"

        assert csv_path.is_file()
        assert jsonl_path.is_file()
        assert summary_path.is_file()
        assert subs_path.is_file()
        assert txn_path.is_file()

        df = pd.read_csv(csv_path)
        assert len(df) == n
        assert df["user_id"].nunique() == n

        subs_df = pd.read_csv(subs_path)
        assert len(subs_df) >= n
        assert "value_score" in subs_df.columns
        assert (subs_df["model_score_grain"] == "subscription").all()

        txn_df = pd.read_csv(txn_path)
        assert len(txn_df) > 0
        assert "value_score" in txn_df.columns
        assert (txn_df["prediction_grain"] == "subscription").all()
        manifest = out / "value_score_eval_manifest.json"
        assert manifest.is_file()
        m = json.loads(manifest.read_text(encoding="utf-8"))
        assert m["score_contract"]["subscription_row_grain"]

        user_csv = pd.read_csv(csv_path)
        assert "base_value_score_user_mean" in user_csv.columns

        lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == n
        first = json.loads(lines[0])
        assert "user_id" in first
        assert "subscriptions_scored" in first

        scores = df["value_score_user_mean"].astype(float)
        assert (scores >= 0).all() and (scores <= _MAX_SCORE).all()

    def test_deterministic_with_fixed_seed(self, tmp_path: Path):
        n = 12
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        _run_script(n, seed=99, outdir=out_a)
        _run_script(n, seed=99, outdir=out_b)
        df_a = pd.read_csv(out_a / f"value_score_eval_{n}.csv")
        df_b = pd.read_csv(out_b / f"value_score_eval_{n}.csv")
        pd.testing.assert_frame_equal(df_a, df_b)


def test_eval_harness_dataset_feature_pipeline():
    ref = datetime(2024, 1, 1, 12, 0, 0)
    data, profile = generate_eval_harness_dataset(
        n_users=8, months_history=4, seed=3, reference_time=ref
    )
    assert len(profile) == 8
    assert set(data.keys()) >= {
        "merchants",
        "subscriptions",
        "transactions",
        "user_explicit",
        "user_computed",
        "user_inferred",
        "user_facts",
        "feedback_signals",
    }
    cfg = load_config()
    cfg["training"]["epochs"] = 1
    cfg["xgboost"]["n_estimators"] = 5
    fe = FeatureEngineer(cfg)
    X, meta = fe.fit_transform(data, reference_time=ref)
    assert len(X) == len(meta)
    subs = data["subscriptions"]
    assert subs.groupby("user_id")["status"].apply(lambda s: (s == "active").any()).all()
