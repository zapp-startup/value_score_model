# Daily scoring (should be good for now)

from __future__ import annotations
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from ..models.value_score_model import ValueScoreModel
from ..config import load_config


def run_daily_scoring(
    model_dir: str | Path,
    data: dict[str, pd.DataFrame],
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
 
    ref = reference_time or datetime.utcnow()
    config = load_config(config_path)

    print(f"\n{'='*55}")
    print(f"  DAILY BATCH SCORING  —  {ref.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*55}")


    subs = data["subscriptions"]
    active_subs = subs[subs["status"] == "active"].copy()
    n_total = len(subs)
    n_active = len(active_subs)
    print(f"  Subscriptions: {n_active} active / {n_total} total")
    print(f"  Users:         {active_subs['user_id'].nunique()}")
    print(f"  Merchants:     {active_subs['merchant_id'].nunique()}")


    scoring_data = dict(data)
    scoring_data["subscriptions"] = active_subs


    print(f"\nLoading model from {model_dir}...")
    model = ValueScoreModel.load(model_dir, config)


    print("Scoring...")
    results = model.predict(scoring_data, reference_time=ref)


    results["scored_at"] = ref.isoformat()


    rec_cfg = config.get("recommendation", {})
    strong_buy = rec_cfg.get("strong_buy_threshold", 85)
    buy = rec_cfg.get("buy_threshold", 65)
    skip = rec_cfg.get("skip_threshold", 35)

    def score_to_recommendation(score: int) -> str:
        if score >= strong_buy:
            return "buy"
        elif score >= buy:
            return "buy"
        elif score <= skip:
            return "skip"
        else:
            return "wait"

    results["recommendation"] = results["value_score"].apply(score_to_recommendation)


    if "evidence_json" in results.columns:
        results["evidence_json"] = results["evidence_json"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )


    print(f"\nScoring complete. {len(results)} scores generated.")
    print(f"  Score distribution:")
    print(f"    Mean:   {results['value_score'].mean():.1f}")
    print(f"    Median: {results['value_score'].median():.1f}")
    print(f"    Min:    {results['value_score'].min()}")
    print(f"    Max:    {results['value_score'].max()}")
    print(f"  Tier breakdown: {results['tier_used'].value_counts().sort_index().to_dict()}")
    print(f"  Recommendations: {results['recommendation'].value_counts().to_dict()}")


    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

    print(f"{'='*55}\n")
    return results


def _load_data_from_dir(data_dir: Path) -> dict[str, pd.DataFrame]:

    keys = [
        "merchants", "subscriptions", "subscription_usage", "transactions",
        "user_explicit", "user_computed", "user_inferred",
        "user_facts", "feedback_signals"
    ]
    data = {}
    for key in keys:
        path = data_dir / f"{key}.csv"
        if path.exists():
            data[key] = pd.read_csv(path)
            print(f"  Loaded {key}: {len(data[key])} rows")
        else:
            print(f"  (skipping {key} — file not found)")
    return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily value score batch job")
    parser.add_argument("--model-dir", required=True, help="Path to saved model directory")
    parser.add_argument("--data-dir", required=True, help="Path to directory with CSV data files")
    parser.add_argument("--output-dir", default="outputs/", help="Where to save results CSV")
    parser.add_argument("--config", default=None, help="Optional config YAML path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"\nLoading data from {data_dir}...")
    data = _load_data_from_dir(data_dir)

    output_path = Path(args.output_dir) / f"scores_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    run_daily_scoring(
        model_dir=args.model_dir,
        data=data,
        config_path=args.config,
        output_path=output_path,
    )