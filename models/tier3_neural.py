#tier 3 nueral net


from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import joblib
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x


class ValueScoreDataset(Dataset):


    def __init__(
        self,
        user_ids: np.ndarray,
        merchant_ids: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        sequences: Optional[np.ndarray] = None,  
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.merchant_ids = torch.LongTensor(merchant_ids)
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) / 100.0 
        self.sequences = torch.FloatTensor(sequences) if sequences is not None else None

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        item = {
            "user_id": self.user_ids[idx],
            "merchant_id": self.merchant_ids[idx],
            "features": self.features[idx],
            "target": self.targets[idx],
        }
        if self.sequences is not None:
            item["sequence"] = self.sequences[idx]
        return item


class ValueScoreNet(nn.Module):

    def __init__(
        self,
        n_users: int,
        n_merchants: int,
        feature_dim: int,
        user_emb_dim: int = 32,
        merchant_emb_dim: int = 16,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        use_gru: bool = True,
        gru_seq_feat_dim: int = 8,
        gru_hidden: int = 32,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        self.user_embedding = nn.Embedding(n_users + 1, user_emb_dim, padding_idx=0)
        self.merchant_embedding = nn.Embedding(n_merchants + 1, merchant_emb_dim, padding_idx=0)

        self.use_gru = use_gru
        if use_gru:
            self.gru = nn.GRU(
                input_size=gru_seq_feat_dim,
                hidden_size=gru_hidden,
                batch_first=True,
            )
            gru_out_dim = gru_hidden
        else:
            gru_out_dim = 0


        mlp_input_dim = user_emb_dim + merchant_emb_dim + feature_dim + gru_out_dim

        layers = []
        prev_dim = mlp_input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        merchant_ids: torch.Tensor,
        features: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u_emb = self.user_embedding(user_ids)           
        m_emb = self.merchant_embedding(merchant_ids)   

        parts = [u_emb, m_emb, features]

        if self.use_gru and sequences is not None:
            _, h_n = self.gru(sequences)               
            gru_out = h_n.squeeze(0)                
            parts.append(gru_out)

        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)                


class NeuralValueModel:

    CONFIDENCE_RANGE = (0.75, 0.95)

    def __init__(self, config: dict):
        self.config = config
        self.net: Optional[ValueScoreNet] = None
        self._user_id_map: dict[int, int] = {}   
        self._merchant_id_map: dict[int, int] = {}
        self._feature_cols: Optional[list[str]] = None
        self._fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        user_ids: np.ndarray,
        merchant_ids: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        user_ids_val: Optional[np.ndarray] = None,
        merchant_ids_val: Optional[np.ndarray] = None,
        sequences: Optional[np.ndarray] = None,
        sequences_val: Optional[np.ndarray] = None,
    ) -> "NeuralValueModel":

        cfg_nn = self.config.get("neural", {})
        cfg_tr = self.config.get("training", {})


        unique_users = sorted(set(user_ids))
        unique_merchants = sorted(set(merchant_ids))
        self._user_id_map = {uid: i + 1 for i, uid in enumerate(unique_users)}
        self._merchant_id_map = {mid: i + 1 for i, mid in enumerate(unique_merchants)}

        self._feature_cols = list(X.columns)
        feature_dim = len(self._feature_cols)

        use_gru = cfg_nn.get("use_gru", True) and sequences is not None
        gru_seq_feat_dim = sequences.shape[-1] if sequences is not None else 8

        self.net = ValueScoreNet(
            n_users=len(unique_users),
            n_merchants=len(unique_merchants),
            feature_dim=feature_dim,
            user_emb_dim=cfg_nn.get("user_embedding_dim", 32),
            merchant_emb_dim=cfg_nn.get("item_embedding_dim", 16),
            hidden_dims=cfg_nn.get("hidden_dims", [128, 64, 32]),
            dropout=cfg_nn.get("dropout", 0.3),
            use_gru=use_gru,
            gru_seq_feat_dim=gru_seq_feat_dim,
            gru_hidden=cfg_nn.get("gru_hidden", 32),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg_tr.get("learning_rate", 0.001),
            weight_decay=cfg_tr.get("weight_decay", 0.0001),
        )
        loss_fn = nn.HuberLoss()

        train_ds = ValueScoreDataset(
            self._map_ids(user_ids, self._user_id_map),
            self._map_ids(merchant_ids, self._merchant_id_map),
            X.fillna(0).values,
            y.values,
            sequences,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg_tr.get("batch_size", 256),
            shuffle=True,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            val_ds = ValueScoreDataset(
                self._map_ids(user_ids_val, self._user_id_map),
                self._map_ids(merchant_ids_val, self._merchant_id_map),
                X_val.fillna(0).values,
                y_val.values,
                sequences_val,
            )
            val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

        patience = cfg_tr.get("early_stopping_patience", 8)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        epochs = cfg_tr.get("epochs", 50)
        print(f"Training Tier 3 neural model for up to {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            self.net.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = self._forward_batch(batch)
                loss = loss_fn(out, batch["target"].to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(batch["target"])
            train_loss /= len(train_ds)

            if has_val:
                val_loss = self._eval_loss(val_loader, loss_fn)
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d} | train={train_loss:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self._fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        user_ids: np.ndarray,
        merchant_ids: np.ndarray,
        sequences: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_fitted()
        self.net.eval()

        ds = ValueScoreDataset(
            self._map_ids(user_ids, self._user_id_map),
            self._map_ids(merchant_ids, self._merchant_id_map),
            self._align_features(X).fillna(0).values,
            np.zeros(len(X)),
            sequences,
        )
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                out = self._forward_batch(batch)
                preds.append(out.cpu().numpy())

        scores = np.concatenate(preds) * 100
        return np.clip(scores, 0, 100).astype(np.float32)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        user_ids: np.ndarray,
        merchant_ids: np.ndarray,
        txn_counts: Optional[np.ndarray] = None,
        sequences: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.predict(X, user_ids, merchant_ids, sequences)
        low, high = self.CONFIDENCE_RANGE
        if txn_counts is not None:
            alpha = np.clip((txn_counts - 50) / 200, 0, 1)
            confidences = low + alpha * (high - low)
        else:
            confidences = np.full(len(scores), np.mean(self.CONFIDENCE_RANGE))
        return scores, confidences.astype(np.float32)

    def save(self, path: str | Path) -> None:
        state = {
            "config": self.config,
            "user_id_map": self._user_id_map,
            "merchant_id_map": self._merchant_id_map,
            "feature_cols": self._feature_cols,
            "net_state": self.net.state_dict() if self.net else None,
            "net_kwargs": self._net_kwargs if hasattr(self, "_net_kwargs") else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, config: dict) -> "NeuralValueModel":
        m = cls(config)
        state = torch.load(path, map_location="cpu")
        m._user_id_map = state["user_id_map"]
        m._merchant_id_map = state["merchant_id_map"]
        m._feature_cols = state["feature_cols"]
        if state["net_state"] and state.get("net_kwargs"):
            m.net = ValueScoreNet(**state["net_kwargs"])
            m.net.load_state_dict(state["net_state"])
            m.net.to(m.device)
        m._fitted = True
        return m


    def _forward_batch(self, batch: dict) -> torch.Tensor:
        uid = batch["user_id"].to(self.device)
        mid = batch["merchant_id"].to(self.device)
        feat = batch["features"].to(self.device)
        seq = batch.get("sequence")
        if seq is not None:
            seq = seq.to(self.device)
        return self.net(uid, mid, feat, seq)

    def _eval_loss(self, loader: DataLoader, loss_fn) -> float:
        self.net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                out = self._forward_batch(batch)
                loss = loss_fn(out, batch["target"].to(self.device))
                total += loss.item() * len(batch["target"])
                n += len(batch["target"])
        return total / max(n, 1)

    def _map_ids(self, ids: np.ndarray, id_map: dict) -> np.ndarray:
        """Map raw IDs to embedding indices. Unknown IDs → 0 (padding)."""
        return np.array([id_map.get(int(i), 0) for i in ids])

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._feature_cols is None:
            return X
        missing = set(self._feature_cols) - set(X.columns)
        X = X.copy()
        for col in missing:
            X[col] = 0.0
        return X[self._feature_cols]

    def _check_fitted(self):
        if not self._fitted or self.net is None:
            raise RuntimeError("Call fit() before predict()")