"""Metrics. Decision thresholds are always chosen on validation data, never on test data."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.nanargmax(f[:-1])]) if len(t) else 0.5


def recall_threshold(y: np.ndarray, s: np.ndarray, recall: float) -> float:
    """Largest threshold that still keeps `recall` of the positives (used for fog forwarding)."""
    pos = np.sort(s[y == 1])
    if len(pos) == 0:
        return float(np.quantile(s, 0.5))
    k = int(np.floor((1 - recall) * len(pos)))
    return float(pos[k])


class ContextThreshold:
    """Context-adaptive threshold: one threshold per (role group, off-hours) context, chosen on
    validation data; contexts with too few validation positives fall back to the role group, then
    to the global threshold."""

    def __init__(self, min_pos: int = 3):
        self.min_pos = min_pos

    def fit(self, y, s, role, off):
        self.glob = best_f1_threshold(y, s)
        self.t = {}
        for r in np.unique(role):
            m = role == r
            t_r = best_f1_threshold(y[m], s[m]) if y[m].sum() >= self.min_pos else self.glob
            self.t[(r, None)] = t_r
            for o in (0.0, 1.0):
                mm = m & (off == o)
                self.t[(r, o)] = best_f1_threshold(y[mm], s[mm]) if y[mm].sum() >= self.min_pos else t_r
        return self

    def predict(self, s, role, off, load: float = 0.0, lam: float = 0.0):
        thr = np.array([self.t.get((r, o), self.t.get((r, None), self.glob)) for r, o in zip(role, off)])
        return (s >= thr + lam * load).astype(int)


def binary_report(y: np.ndarray, s: np.ndarray, pred: np.ndarray) -> dict:
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": fp / max(fp + tn, 1),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()),
    }


def multiclass_report(y, pred, proba=None) -> dict:
    out = {"accuracy": float(accuracy_score(y, pred)),
           "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
           "mcc": float(matthews_corrcoef(y, pred)), "n": int(len(y))}
    if proba is not None and proba.ndim == 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["pr_auc"] = float(average_precision_score(y, proba))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference a-b over seeds with a 95% percentile bootstrap interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
