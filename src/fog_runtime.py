"""Fog-node runtime: loads the exported fog model and scores events. Depends only on numpy,
lightgbm and scikit-learn (no torch), so it runs inside a small resource-limited container."""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np


class FogRuntime:
    def __init__(self, bundle_dir: str | Path):
        d = Path(bundle_dir)
        with open(d / "fog_bundle.pkl", "rb") as f:
            b = pickle.load(f)
        self.feats, self.names = b["feats"], b["fog_names"]
        self.mu, self.sd, self.clip = b["scaler"]["mu"], b["scaler"]["sd"], b["scaler"]["clip"]
        self.iforest, self.ae_w, self.tau_fwd = b["iforest"], b["ae_w"], b["tau_fwd"]
        self.tau_max = b.get("tau_max", self.tau_fwd)
        self.booster = lgb.Booster(model_file=str(d / "fog_lgbm.txt"))

    def _norm(self, X):
        z = np.sign(X) * np.log1p(np.abs(X))
        return np.clip((z - self.mu) / self.sd, -self.clip, self.clip).astype("float32")

    def _ae(self, Xn):
        h = Xn
        for i, (W, b) in enumerate(self.ae_w):
            h = h @ W + b
            if i < len(self.ae_w) - 1:
                h = np.maximum(h, 0)
        return ((h - Xn) ** 2).mean(1)

    def inputs(self, X: np.ndarray) -> np.ndarray:
        """Raw behaviour/context features -> fog model inputs (features + IF + AE scores)."""
        Xn = self._norm(X)
        return np.hstack([X, -self.iforest.score_samples(Xn)[:, None], self._ae(Xn)[:, None]]).astype("float32")

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(self.inputs(X))

    def forward(self, score: np.ndarray) -> np.ndarray:
        """True = send to the cloud for deep correlation."""
        return score >= self.tau_fwd

    def explain(self, X: np.ndarray, top: int = 3) -> list[list[tuple[str, float]]]:
        """Per-alert reasons: LightGBM's built-in TreeSHAP contributions (no extra library)."""
        c = self.booster.predict(self.inputs(X), pred_contrib=True)[:, :-1]
        idx = np.argsort(-np.abs(c), axis=1)[:, :top]
        return [[(self.names[j], float(c[i, j])) for j in row] for i, row in enumerate(idx)]

    def contributions(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(self.inputs(X), pred_contrib=True)[:, :-1]
