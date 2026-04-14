#!/usr/bin/env python3
"""
Stress-test the ValueScoreModel on synthetic users and export audit artifacts.

Uses :func:`data.sample_generator.generate_eval_harness_dataset` plus the real
``ValueScoreModel`` pipeline (feature engineering + tier routing). By default,
if ``--model-dir`` is not set or missing checkpoints, the model is **fit** on
the generated batch (same seed → reproducible data and training init). Use
``--model-dir`` to score with a saved checkpoint instead.

Outputs (under ``--outdir``)::

    {base_name}_{n_users}.csv                — user-level rollup (mean score across subs)
    {base_name}_{n_users}_subscriptions.csv  — **one row per scored subscription** (true model grain)
    {base_name}_{n_users}_transactions.csv — **one row per transaction**, joined to that sub’s scores
    {base_name}_{n_users}.jsonl              — one JSON object per user (includes per-sub detail)
    value_score_eval_summary.txt             — distribution diagnostics (fixed name)

**What the model scores:** ``ValueScoreModel.predict`` outputs one score per **active
subscription** (user + merchant + subscription), using **all** transactions for that
pair aggregated into features. Transaction rows repeat the **same** subscription
scores for audit context — there is no separate per-txn forward pass unless you add one.

Example — full 2,000-user evaluation::

    python value_score_model/scripts/simulate_and_export_value_scores.py \\
        --n-users 2000 --seed 42 --months-history 6 --outdir outputs

Change user count or seed via ``--n-users`` and ``--seed``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: package folder is ``value_score_model``; parent must be on path.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT.parent))

from value_score_model.config import load_config, recommendation_thresholds, scoring_scale
from value_score_model.data.sample_generator import generate_eval_harness_dataset
from value_score_model.models.value_score_model import ValueScoreModel

# Fixed reference time for reproducibility (avoids wall-clock drift).
DEFAULT_REFERENCE_TIME = datetime(2024, 6, 15, 12, 0, 0)

FEATURE_EXPORT_COLS = [
    "cost_weight",
    "quality_weight",
    "sustainability_weight",
    "satisfaction_rating_norm",
    "regret_rating_norm_inv",
    "repurchase_likelihood_norm",
    "usage_decay",
    "time_decay",
    "text_feedback_score",
    "text_feedback_confidence",
    "transaction_count_30d",
    "price_norm",
    "percent_income_on_subs",
    "budget_adherence_score",
    "impulse_susceptibility_score",
    "regret_sensitivity",
]


def score_bucket_label(score: float, max_score: float = 150) -> str:
    s = int(round(float(score)))
    m = int(round(max_score))
    s = max(0, min(s, m))
    if m <= 100:
        if s >= 90:
            return "90-100"
        lo = (s // 10) * 10
        return f"{lo}-{lo + 10}"
    lo = (s // 10) * 10
    hi = min(lo + 10, m)
    if hi >= m:
        return f"{max(0, m - 10)}-{m}"
    return f"{lo}-{hi}"


def _apply_harness_training_caps(config: dict, full_training: bool) -> dict:
    if full_training:
        return config
    c = copy.deepcopy(config)
    c["training"]["epochs"] = min(int(c["training"].get("epochs", 50)), 15)
    c["xgboost"]["n_estimators"] = min(int(c["xgboost"].get("n_estimators", 300)), 120)
    return c


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_or_fit_model(
    config: dict,
    data: dict[str, pd.DataFrame],
    reference_time: datetime,
    model_dir: Path | None,
    full_training: bool,
) -> ValueScoreModel:
    cfg = _apply_harness_training_caps(config, full_training)
    cfg["training"]["random_state"] = cfg["training"].get("random_state", 42)
    cfg["xgboost"]["random_state"] = cfg["xgboost"].get("random_state", 42)

    if model_dir is not None and (model_dir / "meta.pkl").exists():
        return ValueScoreModel.load(model_dir, cfg)

    model = ValueScoreModel(cfg)
    model.fit(data, reference_time=reference_time)
    return model


def _active_scoring_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = dict(data)
    subs = data["subscriptions"].copy()
    out["subscriptions"] = subs[subs["status"] == "active"].reset_index(drop=True)
    return out


def _mode_int(series: pd.Series) -> int:
    c = Counter(series.tolist())
    return int(c.most_common(1)[0][0])


def _attach_meta_columns(results: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    skip = {"user_id", "merchant_id", "subscription_id"}
    for c in meta.columns:
        if c not in skip and c not in out.columns:
            out[c] = meta[c].values
    return out


def _evidence_to_json_str(series: pd.Series) -> pd.Series:
    def _one(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        if isinstance(x, str):
            return x
        try:
            return json.dumps(x, default=_json_safe)
        except TypeError:
            return str(x)

    return series.map(_one)


def build_subscription_export_df(
    results: pd.DataFrame,
    scoring_data: dict[str, pd.DataFrame],
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """One row per model prediction (active subscription)."""
    out = results.copy()
    if "evidence_json" in out.columns:
        out = out.copy()
        out["evidence_json"] = _evidence_to_json_str(out["evidence_json"])

    subs = scoring_data["subscriptions"].rename(columns={"id": "subscription_id"})
    sub_cols = [
        c
        for c in [
            "subscription_id",
            "user_id",
            "merchant_id",
            "price",
            "billing_cycle",
            "status",
            "started_on",
            "renewal_date",
        ]
        if c in subs.columns
    ]
    out = out.merge(
        subs[sub_cols],
        on=["user_id", "merchant_id", "subscription_id"],
        how="left",
    )
    merch = scoring_data["merchants"].rename(columns={"id": "merchant_id"})
    mcols = [c for c in ["merchant_id", "name", "category"] if c in merch.columns]
    out = out.merge(merch[mcols], on="merchant_id", how="left")
    if "persona" in profile.columns:
        out = out.merge(profile[["user_id", "persona"]], on="user_id", how="left")
    out.insert(0, "model_score_grain", "subscription")
    return out


def build_transaction_audit_df(
    results: pd.DataFrame,
    scoring_data: dict[str, pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    """
    One row per transaction; model scores are **subscription-level** (repeated for each txn).
    """
    tx = scoring_data["transactions"].copy()
    keys = ["user_id", "merchant_id", "subscription_id"]
    score_cols = [
        c
        for c in [
            "user_id",
            "merchant_id",
            "subscription_id",
            "value_score",
            "display_value_score",
            "base_value_score",
            "overutilisation_bonus",
            "tier_used",
            "confidence",
        ]
        if c in results.columns
    ]
    sub_scores = results[score_cols].drop_duplicates(subset=keys)
    audit = tx.merge(sub_scores, on=keys, how="left")
    scales = scoring_scale(config)
    col, _thr = recommendation_thresholds(config)
    audit.insert(0, "prediction_grain", "subscription")
    audit.insert(1, "learned_score_max", scales["learned_max"])
    audit.insert(2, "display_score_max", scales["display_max"])
    audit.insert(
        3,
        "subscription_score_audit_note",
        "Scores are computed once per active subscription from aggregated history; "
        "this row repeats that subscription score for txn-level audit. "
        "Not an independent transaction model prediction. "
        f"Recommendations config uses score_column={col}.",
    )
    return audit


def run_export(
    n_users: int,
    seed: int,
    months_history: int,
    outdir: Path,
    base_name: str,
    reference_time: datetime,
    model_dir: Path | None,
    full_training: bool,
    config_path: str | None,
) -> tuple[pd.DataFrame, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    _set_global_seeds(seed)

    config = load_config(config_path)
    data, profile = generate_eval_harness_dataset(
        n_users=n_users,
        months_history=months_history,
        seed=seed,
        reference_time=reference_time,
    )
    scoring_data = _active_scoring_data(data)

    model = _load_or_fit_model(
        config, data, reference_time, model_dir, full_training
    )
    results = model.predict(scoring_data, reference_time=reference_time)
    X, meta = model.feature_engineer.transform(scoring_data, reference_time)

    results = results.reset_index(drop=True)
    X = X.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    results = _attach_meta_columns(results, meta)
    feat_cols = [c for c in FEATURE_EXPORT_COLS if c in X.columns]
    for c in feat_cols:
        results[c] = X[c].values

    scales = scoring_scale(config)
    max_score = scales["display_max"]
    learned_max = scales["learned_max"]

    subs_csv_path = outdir / f"{base_name}_{n_users}_subscriptions.csv"
    txn_csv_path = outdir / f"{base_name}_{n_users}_transactions.csv"
    manifest_path = outdir / f"{base_name}_manifest.json"
    sub_export_df = build_subscription_export_df(results, scoring_data, profile)
    sub_export_df.to_csv(subs_csv_path, index=False)
    txn_audit_df = build_transaction_audit_df(results, scoring_data, config)
    txn_audit_df.to_csv(txn_csv_path, index=False)

    user_rows = []
    jsonl_lines: list[str] = []
    jsonl_path = outdir / f"{base_name}_{n_users}.jsonl"
    csv_path = outdir / f"{base_name}_{n_users}.csv"

    for uid, g in results.groupby("user_id"):
        uid = int(uid)
        mean_score = float(g["value_score"].mean())
        mean_base = float(g["base_value_score"].mean())
        tier_mode = _mode_int(g["tier_used"])
        conf_mean = float(g["confidence"].mean())
        feat_mean = {c: float(g[c].mean()) for c in feat_cols}

        prof = profile[profile["user_id"] == uid]
        prof_d = prof.iloc[0].to_dict() if len(prof) else {}

        ue = data["user_explicit"][data["user_explicit"]["user_id"] == uid].iloc[0].to_dict()
        uc = data["user_computed"][data["user_computed"]["user_id"] == uid].iloc[0].to_dict()
        ui = data["user_inferred"][data["user_inferred"]["user_id"] == uid].iloc[0].to_dict()

        subs_detail = []
        for _, row in g.iterrows():
            ev = row.get("evidence_json")
            if hasattr(ev, "item"):
                ev = ev.item()
            subs_detail.append(
                {
                    "merchant_id": int(row["merchant_id"]),
                    "subscription_id": int(row["subscription_id"]),
                    "value_score": int(row["value_score"]),
                    "base_value_score": int(row["base_value_score"])
                    if "base_value_score" in row
                    else None,
                    "overutilisation_bonus": float(row["overutilisation_bonus"])
                    if "overutilisation_bonus" in row
                    else None,
                    "confidence": float(row["confidence"]),
                    "tier_used": int(row["tier_used"]),
                    "evidence": ev,
                    "features": {c: float(row[c]) for c in feat_cols},
                }
            )

        flat = {
            "user_id": uid,
            "value_score_user_mean": round(mean_score, 2),
            "base_value_score_user_mean": round(mean_base, 2),
            "score_bucket": score_bucket_label(mean_score, max_score),
            "base_score_bucket": score_bucket_label(mean_base, learned_max),
            "tier_used_mode": tier_mode,
            "confidence_mean": round(conf_mean, 4),
            "n_subscriptions_scored": len(g),
            "persona": prof_d.get("persona"),
            "months_history": prof_d.get("months_history"),
            "credit_stress_proxy": prof_d.get("credit_stress_proxy"),
            "overdraft_risk_proxy": prof_d.get("overdraft_risk_proxy"),
            "txn_count_horizon": prof_d.get("txn_count_horizon"),
            "avg_monthly_txn_count": prof_d.get("avg_monthly_txn_count"),
            "category_diversity": prof_d.get("category_diversity"),
            "essentials_txn_share": prof_d.get("essentials_txn_share"),
            "discretionary_txn_share": prof_d.get("discretionary_txn_share"),
            "transfer_style_proxy": prof_d.get("transfer_style_proxy"),
            "cash_withdrawal_proxy": prof_d.get("cash_withdrawal_proxy"),
            "monthly_income": ue.get("monthly_income"),
            "monthly_fixed_expenses": ue.get("monthly_fixed_expenses"),
            "life_stage": ue.get("life_stage"),
            "financial_goal": ue.get("financial_goal"),
            "actual_monthly_spending": ui.get("actual_monthly_spending"),
            "percent_income_spent_on_subscriptions": ui.get(
                "percent_income_spent_on_subscriptions"
            ),
            "budget_adherence_score": uc.get("budget_adherence_score"),
            "impulse_susceptibility_score": uc.get("impulse_susceptibility_score"),
        }
        for k, v in feat_mean.items():
            flat[f"feat_mean_{k}"] = round(v, 6) if v == v else None

        user_rows.append(flat)

        record = {
            "user_id": uid,
            "model_score_grain": "subscription",
            "model_score_scale": (
                f"learned base 0..{int(learned_max)}; display (value_score) 0..{int(max_score)} "
                "= base + overutilisation_bonus (clipped)"
            ),
            "profile_generation": prof_d,
            "user_explicit": {k: _json_safe(v) for k, v in ue.items()},
            "user_computed": {k: _json_safe(v) for k, v in uc.items()},
            "user_inferred": {k: _json_safe(v) for k, v in ui.items()},
            "value_score_user_mean": round(mean_score, 4),
            "base_value_score_user_mean": round(mean_base, 4),
            "score_bucket": score_bucket_label(mean_score, max_score),
            "base_score_bucket": score_bucket_label(mean_base, learned_max),
            "tier_used_mode": tier_mode,
            "confidence_mean": round(conf_mean, 4),
            "feature_means": {k: round(v, 6) for k, v in feat_mean.items()},
            "subscriptions_scored": subs_detail,
        }
        jsonl_lines.append(json.dumps(record, default=_json_safe))

    user_df = pd.DataFrame(user_rows).sort_values("user_id").reset_index(drop=True)
    if len(user_df) != n_users:
        raise RuntimeError(
            f"Expected {n_users} scored users, got {len(user_df)} "
            "(check subscriptions / active filter)."
        )
    user_df.to_csv(csv_path, index=False)
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")

    scores = user_df["value_score_user_mean"].values
    base_user = user_df["base_value_score_user_mean"].values
    stats = _compute_distribution_stats(
        scores, n_users, max_score=max_score, label_prefix="user_mean_display_value_score"
    )
    stats_base = _compute_distribution_stats(
        base_user, n_users, max_score=learned_max, label_prefix="user_mean_base_value_score"
    )
    summary_path = outdir / "value_score_eval_summary.txt"
    rec_col, _ = recommendation_thresholds(config)
    _write_manifest(
        manifest_path,
        config,
        rec_col,
        scales,
        {
            "user_csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "subscriptions_csv": str(subs_csv_path),
            "transactions_csv": str(txn_csv_path),
        },
        n_users,
        len(results),
        len(txn_audit_df),
    )
    _write_summary(
        summary_path,
        stats,
        stats_base,
        csv_path,
        jsonl_path,
        subs_csv_path,
        txn_csv_path,
        manifest_path,
        max_score,
        learned_max,
    )

    print(stats["report_text"])
    print(stats_base["report_text"])
    return user_df, stats


def _json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if x == x else None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.isoformat()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def _compute_distribution_stats(
    scores: np.ndarray,
    n_users: int,
    max_score: float = 150,
    label_prefix: str = "metric",
) -> dict:
    s = np.asarray(scores, dtype=float)
    m = int(round(max_score))
    hist: dict[str, int] = {}
    lines = []
    lines.append(f"{label_prefix}_scale_0..{m}")
    lines.append(f"users_requested={n_users}")
    lines.append(f"users_scored={len(s)}")
    lines.append(f"mean={float(np.mean(s)):.4f}")
    lines.append(f"median={float(np.median(s)):.4f}")
    lines.append(f"std={float(np.std(s)):.4f}")
    lines.append(f"min={float(np.min(s)):.4f}")
    lines.append(f"max={float(np.max(s)):.4f}")
    pct_60_70 = float(np.mean((s >= 60) & (s <= 70)) * 100)
    lines.append(f"pct_scores_in_60_70_inclusive={pct_60_70:.2f}%")
    lines.append(f"count_lt_20={int((s < 20).sum())}")
    lines.append(f"count_gt_90={int((s > 90).sum())}")
    lines.append("histogram_counts_by_bucket_label:")
    for lo in range(0, m, 10):
        if lo + 10 >= m:
            label = f"{lo}-{m}"
            c = int(((s >= lo) & (s <= m)).sum())
        else:
            label = f"{lo}-{lo + 10}"
            c = int(((s >= lo) & (s < lo + 10)).sum())
        hist[label] = c
        lines.append(f"  {label}: {c}")
    report = "\n".join(lines) + "\n"
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "pct_60_70": pct_60_70,
        "count_lt_20": int((s < 20).sum()),
        "count_gt_90": int((s > 90).sum()),
        "histogram": hist,
        "report_text": report,
    }


def _write_manifest(
    path: Path,
    config: dict,
    recommendation_score_column: str,
    scales: dict[str, float],
    output_paths: dict[str, str],
    n_users: int,
    n_subscription_scores: int,
    n_transaction_rows: int,
) -> None:
    doc = {
        "score_contract": {
            "learned_score_max": scales["learned_max"],
            "display_max": scales["display_max"],
            "value_score_definition": "base_value_score + overutilisation_bonus clipped to display_max",
            "subscription_row_grain": "one prediction per active subscription",
            "transaction_csv_grain": "subscription scores repeated per transaction row; not TransactionValueModel",
            "recommendation_score_column": recommendation_score_column,
        },
        "outputs": output_paths,
        "counts": {
            "n_users_requested": n_users,
            "n_subscription_score_rows": n_subscription_scores,
            "n_transaction_export_rows": n_transaction_rows,
        },
        "phase_b": "Train per-transaction model via scripts/train_transaction_value.py",
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_summary(
    path: Path,
    stats: dict,
    stats_base: dict,
    csv_path: Path,
    jsonl_path: Path,
    subs_csv_path: Path,
    txn_csv_path: Path,
    manifest_path: Path,
    max_score: float,
    learned_max: float,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Display / combined (user mean of value_score) ===\n")
        f.write(stats["report_text"])
        f.write("\n=== Learned base only (user mean of base_value_score) ===\n")
        f.write(stats_base["report_text"])
        f.write(f"csv_user_level={csv_path}\n")
        f.write(f"jsonl_user_level={jsonl_path}\n")
        f.write(f"csv_subscriptions_one_row_per_score={subs_csv_path}\n")
        f.write(f"csv_transactions_with_subscription_scores={txn_csv_path}\n")
        f.write(f"manifest={manifest_path}\n")
        f.write(
            f"note=Subscription CSV: learned base 0..{int(learned_max)}, "
            f"display value_score 0..{int(max_score)}. "
            "Txn CSV: prediction_grain=subscription (repeated). "
            "See manifest for contract.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate synthetic users and export value score evaluation artifacts."
    )
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--months-history", type=int, default=6)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--base-name",
        type=str,
        default="value_score_eval",
        help="Output stem: {base_name}_{n_users}.csv / .jsonl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with meta.pkl + tier artifacts; if missing, fit on synthetic data.",
    )
    parser.add_argument(
        "--full-training",
        action="store_true",
        help="Use full config epochs/trees (slow). Default caps training for harness speed.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    args = parser.parse_args()

    stem = f"{args.base_name}_{args.n_users}"
    for path in [
        args.outdir / f"{stem}.jsonl",
        args.outdir / f"{stem}.csv",
        args.outdir / f"{stem}_subscriptions.csv",
        args.outdir / f"{stem}_transactions.csv",
        args.outdir / f"{args.base_name}_manifest.json",
    ]:
        if path.exists():
            path.unlink()

    run_export(
        n_users=args.n_users,
        seed=args.seed,
        months_history=args.months_history,
        outdir=args.outdir,
        base_name=args.base_name,
        reference_time=DEFAULT_REFERENCE_TIME,
        model_dir=args.model_dir,
        full_training=args.full_training,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
