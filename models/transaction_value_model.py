"""
Per-transaction value regressor (Phase B). Separate from subscription ValueScoreModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


class TransactionValueModel:
    """XGBoost regressor on txn-level features; target 0–learned_max (default 100)."""

    def __init__(self, config: Optional[dict] = None):
        from ..config import load_config

        self.config = config or load_config()
        xgb_cfg = self.config.get("transaction_xgboost", self.config.get("xgboost", {}))
        self._learned_max = float(
            self.config.get("scoring", {}).get("learned_score_max", 100)
        )
        self.model = xgb.XGBRegressor(
            n_estimators=int(xgb_cfg.get("n_estimators", 200)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            min_child_weight=int(xgb_cfg.get("min_child_weight", 3)),
            reg_alpha=float(xgb_cfg.get("reg_alpha", 0.1)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
            random_state=int(xgb_cfg.get("random_state", 42)),
            objective="reg:squarederror",
            verbosity=0,
        )
        self._feature_names: Optional[list[str]] = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransactionValueModel":
        self._feature_names = list(X.columns)
        self.model.fit(X.fillna(0).values, y.values)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        cols = self._feature_names or list(X.columns)
        Xa = X.reindex(columns=cols, fill_value=0).fillna(0).values
        raw = self.model.predict(Xa)
        return np.clip(raw, 0, self._learned_max).astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self._feature_names,
                "learned_max": self._learned_max,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransactionValueModel":
        state = joblib.load(path)
        m = cls(state.get("config"))
        m.model = state["model"]
        m._feature_names = state["feature_names"]
        m._learned_max = float(state.get("learned_max", 100))
        m._fitted = True
        return m
