#tier 2 xgboost

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from typing import Optional
import joblib
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - exercised in environments without xgboost
    xgb = None


BLEND_WINDOW = 30 


class XGBoostValueModel:

    CONFIDENCE_RANGE = (0.5, 0.80)

    def __init__(self, config: dict):
        self.config = config
        xgb_cfg = config.get("xgboost", {})
        if xgb is not None:
            self.model = xgb.XGBRegressor(
                n_estimators=xgb_cfg.get("n_estimators", 300),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                min_child_weight=xgb_cfg.get("min_child_weight", 5),
                reg_alpha=xgb_cfg.get("reg_alpha", 0.1),
                reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                random_state=xgb_cfg.get("random_state", 42),
                objective="reg:squarederror",
                verbosity=0,
            )
            self.backend = "xgboost"
        else:
            self.model = HistGradientBoostingRegressor(
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.05),
                max_iter=xgb_cfg.get("n_estimators", 300),
                l2_regularization=xgb_cfg.get("reg_lambda", 1.0),
                random_state=xgb_cfg.get("random_state", 42),
            )
            self.backend = "sklearn_hist_gradient_boosting"
        self._feature_names: Optional[list[str]] = None
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "XGBoostValueModel":
        self._feature_names = list(X.columns)
        X_arr = X.fillna(0).values
        y_arr = y.values

        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val.fillna(0).values, y_val.values)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_arr, y_arr, **fit_kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_aligned = self._align_features(X)
        raw = self.model.predict(X_aligned.fillna(0).values)
        return np.clip(raw, 0, 100).astype(np.float32)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        txn_counts: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.predict(X)

        if txn_counts is not None:

            alpha = np.clip(txn_counts / BLEND_WINDOW, 0, 1)
            low, high = self.CONFIDENCE_RANGE
            confidences = low + alpha * (high - low)
        else:
            confidences = np.full(len(scores), np.mean(self.CONFIDENCE_RANGE))

        return scores, confidences.astype(np.float32)

    def get_feature_importance(self) -> pd.Series:
        self._check_fitted()
        if not hasattr(self.model, "feature_importances_"):
            n_features = len(self._feature_names or [])
            if n_features == 0:
                return pd.Series(dtype=float)
            return pd.Series(
                np.full(n_features, 1.0 / n_features, dtype=float),
                index=self._feature_names,
            ).sort_values(ascending=False)
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostValueModel":
        return joblib.load(path)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:

        if self._feature_names is None:
            return X
        missing = set(self._feature_names) - set(X.columns)
        for col in missing:
            X = X.copy()
            X[col] = 0.0
        return X[self._feature_names]
