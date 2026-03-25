from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(
    predictions: pd.DataFrame,
    ground_truth: pd.Series,
    verbose: bool = True,
) -> dict:
    
    if "base_value_score" in predictions.columns:
        y_pred = predictions["base_value_score"].values.astype(float)
    else:
        y_pred = np.clip(predictions["value_score"].values.astype(float), 0, 100)
    y_true = ground_truth.values.astype(float)

    results = {}

    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    results["mean_pred"] = float(y_pred.mean())
    results["mean_true"] = float(y_true.mean())

    pred_for_rank = predictions
    if "base_value_score" in predictions.columns:
        pred_for_rank = predictions.assign(value_score=predictions["base_value_score"])

    if "user_id" in predictions.columns:
        user_spearman = _per_user_spearman(pred_for_rank, y_true)
        results["mean_spearman"] = float(np.nanmean(user_spearman))
        results["median_spearman"] = float(np.nanmedian(user_spearman))
        results["pct_positive_rank_corr"] = float((user_spearman > 0).mean())
    else:
        corr, _ = spearmanr(y_true, y_pred)
        results["spearman"] = float(corr)

    if "user_id" in predictions.columns:
        for k in [3, 5]:
            results[f"ndcg@{k}"] = _mean_ndcg_at_k(pred_for_rank, y_true, k)

    if "tier_used" in predictions.columns:
        tier_counts = predictions["tier_used"].value_counts().sort_index()
        results["tier_distribution"] = tier_counts.to_dict()

    if "confidence" in predictions.columns:
        results["confidence_calibration"] = _confidence_calibration(
            y_pred, y_true, predictions["confidence"].values
        )

    if verbose:
        _print_results(results)

    return results


def _per_user_spearman(predictions: pd.DataFrame, y_true: np.ndarray) -> np.ndarray:
    pred_df = predictions.copy()
    pred_df["_true"] = y_true
    correlations = []
    for uid, grp in pred_df.groupby("user_id"):
        if len(grp) < 2:
            continue
        corr, _ = spearmanr(grp["value_score"], grp["_true"])
        if not np.isnan(corr):
            correlations.append(corr)
    return np.array(correlations) if correlations else np.array([0.0])


def _mean_ndcg_at_k(predictions: pd.DataFrame, y_true: np.ndarray, k: int) -> float:
    pred_df = predictions.copy()
    pred_df["_true"] = y_true
    ndcgs = []
    for uid, grp in pred_df.groupby("user_id"):
        if len(grp) < 2:
            continue
        ndcg = _ndcg_at_k(grp["value_score"].values, grp["_true"].values, k)
        ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _ndcg_at_k(pred_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    k = min(k, len(pred_scores))
    ideal_order = np.argsort(-true_scores)[:k]
    ideal_gains = true_scores[ideal_order]
    ideal_dcg = np.sum(ideal_gains / np.log2(np.arange(2, k + 2)))

    if ideal_dcg == 0:
        return 0.0

    pred_order = np.argsort(-pred_scores)[:k]
    pred_gains = true_scores[pred_order]
    pred_dcg = np.sum(pred_gains / np.log2(np.arange(2, k + 2)))

    return float(pred_dcg / ideal_dcg)


def _confidence_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    confidences: np.ndarray,
) -> dict:
    df = pd.DataFrame({
        "pred": y_pred, "true": y_true, "conf": confidences
    })
    try:
        df["quartile"] = pd.qcut(
            df["conf"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
    except ValueError:
        return {"all": round(float(mean_absolute_error(df["true"], df["pred"])), 3)}
    cal = df.groupby("quartile", observed=True).apply(
        lambda g: mean_absolute_error(g["true"], g["pred"]), include_groups=False
    ).to_dict()
    return {str(k): round(float(v), 3) for k, v in cal.items()}


def _print_results(results: dict) -> None:
    print("\n" + "=" * 50)
    print("  VALUE SCORE MODEL — EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MAE:                    {results.get('mae', 'N/A'):.3f}")
    print(f"  RMSE:                   {results.get('rmse', 'N/A'):.3f}")
    print(f"  Mean predicted score:   {results.get('mean_pred', 'N/A'):.2f}")
    print(f"  Mean true score:        {results.get('mean_true', 'N/A'):.2f}")
    if "mean_spearman" in results:
        print(f"  Mean Spearman (rank):   {results['mean_spearman']:.4f}")
        print(f"  Median Spearman:        {results['median_spearman']:.4f}")
        print(f"  % positive rank corr:   {results['pct_positive_rank_corr']*100:.1f}%")
    if "ndcg@3" in results:
        print(f"  NDCG@3:                 {results['ndcg@3']:.4f}")
        print(f"  NDCG@5:                 {results['ndcg@5']:.4f}")
    if "tier_distribution" in results:
        print(f"  Tier distribution:      {results['tier_distribution']}")
    if "confidence_calibration" in results:
        print(f"  Confidence calibration (MAE per quartile):")
        for q, mae in results["confidence_calibration"].items():
            print(f"    {q}: MAE={mae:.3f}")
    print("=" * 50 + "\n")