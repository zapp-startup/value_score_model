#cold start model (tier 1)

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
import joblib
from pathlib import Path


class ColdStartModel:
    CONFIDENCE_RANGE = (0.25, 0.45)

    def __init__(self, config: dict):
        self.config = config
        self._population_table: Optional[pd.DataFrame] = None
        self._global_mean: float = 50.0

    def fit(self, base_df: pd.DataFrame, targets: pd.Series) -> "ColdStartModel":
        df = base_df.copy()
        df["_target"] = targets.values

        lvl1 = (
            df.groupby(["life_stage", "financial_goal", "merchant_category"])["_target"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_score", "count": "n"})
        )

        lvl2 = (
            df.groupby(["life_stage", "financial_goal"])["_target"]
            .mean()
            .reset_index()
            .rename(columns={"_target": "avg_score_l2"})
        )

        lvl3 = (
            df.groupby(["life_stage"])["_target"]
            .mean()
            .reset_index()
            .rename(columns={"_target": "avg_score_l3"})
        )

        self._population_table = lvl1
        self._lvl2 = lvl2
        self._lvl3 = lvl3
        self._global_mean = float(df["_target"].mean())

        return self

    def predict(self, base_df: pd.DataFrame) -> np.ndarray:
        """Return array of scores in 0-100 range."""
        if self._population_table is None:
            return np.full(len(base_df), self._global_mean)

        scores = []
        for _, row in base_df.iterrows():
            score = self._lookup_score(row)
            score = self._blend_with_user_weights(score, row)
            scores.append(np.clip(score, 0, 100))

        return np.array(scores, dtype=np.float32)

    def predict_with_confidence(
        self, base_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, confidences) both shape (n,)."""
        scores = self.predict(base_df)


        confidences = np.full(len(scores), np.mean(self.CONFIDENCE_RANGE))
        return scores, confidences

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ColdStartModel":
        return joblib.load(path)


    def _lookup_score(self, row: pd.Series) -> float:
        life_stage = row.get("life_stage", "other")
        fin_goal = row.get("financial_goal", "other")
        category = row.get("merchant_category", "other")


        mask = (
            (self._population_table["life_stage"] == life_stage) &
            (self._population_table["financial_goal"] == fin_goal) &
            (self._population_table["merchant_category"] == category)
        )
        if mask.any():
            return float(self._population_table.loc[mask, "avg_score"].iloc[0])


        mask2 = (
            (self._lvl2["life_stage"] == life_stage) &
            (self._lvl2["financial_goal"] == fin_goal)
        )
        if mask2.any():
            return float(self._lvl2.loc[mask2, "avg_score_l2"].iloc[0])


        mask3 = self._lvl3["life_stage"] == life_stage
        if mask3.any():
            return float(self._lvl3.loc[mask3, "avg_score_l3"].iloc[0])

        return self._global_mean

    def _blend_with_user_weights(self, population_score: float, row: pd.Series) -> float:

        cost_w = row.get("cost_weight", 0.33) or 0.33
        qual_w = row.get("quality_weight", 0.33) or 0.33


        price_norm = row.get("price_norm", 0.05) or 0.05
        price_signal = (1.0 - np.clip(price_norm, 0, 1)) * 100


        category_quality = {
            "streaming": 70, "fitness": 65, "software": 72,
            "education": 68, "grocery": 60, "utilities": 55,
            "food": 62, "other": 55,
        }.get(row.get("merchant_category", "other"), 55)

        weight_signal = cost_w * price_signal + qual_w * category_quality

        blended = (
            0.60 * population_score +
            0.25 * price_signal +
            0.15 * weight_signal
        )
        return float(blended)