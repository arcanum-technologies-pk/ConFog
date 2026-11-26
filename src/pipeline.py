"""The proposed fog-cloud context-aware insider detector, and reproductions of base papers [11], [13].

Proposed (Phases II-III):
  fog   : Isolation Forest + autoencoder anomaly scores (out-of-fold on training rows)
          -> LightGBM on behaviour + deviation + context + anomaly scores
  cloud : GRU over the last 14 user-days + graph learning on the temporal user-PC graph
          (only events the fog forwards are re-scored)
  decision: context-adaptive threshold (role group x off-hours), chosen on validation data
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .cert_features import feature_groups, modality_of
from .common import log, save_pickle
from .metrics import ContextThreshold, best_f1_threshold, recall_threshold
from .graph import graph_inputs
from .models import FogBlend  # PATCH_V3_IMPORT
from .models import AutoEncoder, CloudGRU, FogLGBM, IForest, LSTMAE, NeuralScaler, SeqData

ANOMALY_FREE = {"fog_no_anomaly", "cloud_only", "cloud_only_no_graph"}


def select_features(df: pd.DataFrame, variant: str = "full", context: bool = True) -> list[str]:
    g = feature_groups(df)
    cols = g["behaviour"] + g["deviation"] + (g["context"] if context and variant != "no_context" else [])
    if variant.startswith("drop_"):
        mod = variant[5:]
        cols = [c for c in cols if modality_of(c) != mod]
    if variant == "no_tenure":          # PATCH_REVIEW_FEATS
        cols = [c for c in cols if c != "ctx_tenure"]
    if variant == "no_web_keywords":
        cols = [c for c in cols if not any(k in c for k in ("http_job", "http_leak", "http_hack"))]
    return cols


def _user_folds(users: np.ndarray, y: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k folds over users (no user in two folds); insider users are spread round-robin over the folds."""
    rng = np.random.default_rng(seed)
    pos_users = rng.permutation(np.unique(users[y == 1]))
    other = rng.permutation(np.setdiff1d(np.unique(users), pos_users))
    fold = {u: i % k for i, u in enumerate(pos_users)} | {u: i % k for i, u in enumerate(other)}
    f = np.array([fold[u] for u in users])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def _minmax(train_scores: np.ndarray):
    lo, hi = np.percentile(train_scores, [1, 99])
    return lambda s: np.clip((s - lo) / max(hi - lo, 1e-12), 0, 1)


# ============================================================================ proposed framework
class ProposedFramework:
    def __init__(self, cfg: dict, seed: int, device, variant: str = "full", ckpt_dir: Path | None = None):
        self.cfg, self.seed, self.dev, self.v, self.ckpt = cfg, seed, device, variant, ckpt_dir
        self.use_fog = not variant.startswith("cloud_only")
        self.use_cloud = variant not in ("fog_only", "if_only", "ae_only")
        self.use_anom = variant not in ANOMALY_FREE
        self.unsup_only = variant in ("if_only", "ae_only")
        self.use_graph = "no_graph" not in variant

    def _seqdata(self, df: pd.DataFrame, Xn: np.ndarray) -> SeqData:
        G = np.hstack([Xn, graph_inputs(self.cfg, df.attrs["release"], Xn)]) if self.use_graph else None
        return SeqData(Xn, df["user"].values, self.cfg["models"]["gru"]["window"], self.dev, G)

    # ---------------------------------------------------------------- helpers
    def _anomaly(self, Xn: np.ndarray, y: np.ndarray, users: np.ndarray, tr, va) -> np.ndarray:
        """Anomaly scores for all rows. Training rows get out-of-fold scores (models never saw them)."""
        k = self.cfg["models"]["oof_folds"]
        out = np.zeros((len(Xn), 2), dtype="float32")
        va_norm = Xn[va][y[va] == 0]
        tr_users = np.unique(users[tr])
        fold_of = dict(zip(tr_users, np.random.default_rng(self.seed).integers(0, k, len(tr_users))))
        fold = np.array([fold_of[u] for u in users[tr]])
        for f in range(k):
            fit_rows, score_rows = tr[(fold != f)], tr[fold == f]
            fit_rows = fit_rows[y[fit_rows] == 0]
            out[score_rows] = self._fit_score(Xn[fit_rows], va_norm, Xn[score_rows], f"oof{f}")
        fit_rows = tr[y[tr] == 0]
        rest = np.setdiff1d(np.arange(len(Xn)), tr)
        out[rest] = self._fit_score(Xn[fit_rows], va_norm, Xn[rest], "final", keep=True)
        return out

    def _fit_score(self, Xfit, Xva_norm, Xscore, tag, keep=False):
        ifo = IForest(self.cfg, self.seed).fit(Xfit)
        ck = self.ckpt / f"ae_{tag}.pt" if self.ckpt else None
        ae = AutoEncoder(self.cfg, self.seed, self.dev, tag).fit(Xfit, Xva_norm, ck)
        if keep:
            self.iforest, self.ae = ifo, ae
        return np.column_stack([ifo.score(Xscore), ae.score(Xscore)])

    def _anomaly_apply(self, Xn):
        return np.column_stack([self.iforest.score(Xn), self.ae.score(Xn)]).astype("float32")

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, sp: dict) -> "ProposedFramework":
        t0 = time.perf_counter()
        self.feats = select_features(df, self.v)
        X = df[self.feats].values.astype("float32")
        y = df["label"].values.astype(int)
        users = df["user"].values
        tr, va = sp["train"], sp["val"]
        self.scaler = NeuralScaler(self.cfg["models"]["gru"]["clip"]).fit(X[tr])
        Xn = self.scaler.transform(X)
        log.info(f"  [{self.v}] {len(self.feats)} features | train {len(tr):,} ({y[tr].sum()} pos) | "
                 f"val {len(va):,} ({y[va].sum()} pos)")
        # too few validation positives to select models on (e.g. CERT r6.2, 44 positive user-days in total)
        scarce = y[va].sum() < self.cfg["models"]["min_val_pos_select"]
        if scarce:
            log.info(f"  only {y[va].sum()} positive validation days: trees chosen by grouped CV on train, "
                     f"GRU trained for a fixed number of epochs, thresholds/blend on out-of-fold + validation scores")
            folds = _user_folds(users[tr], y[tr], self.cfg["models"]["oof_folds"], self.seed)

        fog, self.diag = None, {}
        if self.use_anom:
            anom = self._anomaly(Xn, y, users, tr, va)
            log.info("  fog: anomaly detectors ready (out-of-fold on train)")
        if self.unsup_only:
            j = 0 if self.v == "if_only" else 1
            self.norm = _minmax(anom[tr, j][y[tr] == 0])
            fog = self.norm(anom[:, j])
        elif self.use_fog:
            Xf = np.hstack([X, anom]) if self.use_anom else X
            self.fog_names = self.feats + (["if_score", "ae_score"] if self.use_anom else [])
            self.lgbm = (FogBlend if getattr(self, "v3", False) else FogLGBM)(self.cfg["models"]["fog_lgbm"], self.seed)  # PATCH_V3_USE
            if scarce:
                oof = self.lgbm.fit_cv(Xf[tr], y[tr], folds, self.fog_names)
            else:
                self.lgbm.fit(Xf[tr], y[tr], Xf[va], y[va], self.fog_names)
            fog = self.lgbm.score(Xf)
            self.diag["fog_train_pr_auc"] = float(average_precision_score(y[tr], fog[tr]))
            self.diag["fog_val_pr_auc"] = float(average_precision_score(y[va], fog[va]))
            log.info(f"  fog: LightGBM PR-AUC train {self.diag['fog_train_pr_auc']:.4f} | "
                     f"val {self.diag['fog_val_pr_auc']:.4f}")

        cloud = None
        if self.use_cloud:
            data = self._seqdata(df, Xn)
            ck = self.ckpt / "gru.pt" if self.ckpt else None
            self.gru = CloudGRU(self.cfg, self.seed, self.dev).fit(data, y, tr, va, ck, select_on_val=not scarce)
            cloud = np.zeros(len(df), dtype="float32")
            cloud[va] = self.gru.score(data, va)
            sub = np.concatenate([tr[y[tr] == 1], np.random.default_rng(self.seed).choice(
                tr[y[tr] == 0], min(len(va), (y[tr] == 0).sum()), replace=False)])
            self.diag["cloud_train_pr_auc_subsample"] = float(average_precision_score(y[sub], self.gru.score(data, sub)))
            self.diag["cloud_val_pr_auc"] = float(average_precision_score(y[va], cloud[va]))
            if scarce and fog is not None:     # out-of-fold cloud scores -> blend weight not chosen on 1-9 positives
                oof_cloud = np.zeros(len(tr), dtype="float32")
                for i, (a, b) in enumerate(folds):
                    ckf = self.ckpt / f"gru_oof{i}.pt" if self.ckpt else None
                    g = CloudGRU(self.cfg, self.seed, self.dev).fit(data, y, tr[a], va, ckf, select_on_val=False)
                    oof_cloud[b] = g.score(data, tr[b])
                self.diag["cloud_oof_pr_auc"] = float(average_precision_score(y[tr], oof_cloud))
            log.info(f"  cloud: {'GRU+graph' if self.use_graph else 'GRU'} PR-AUC val {self.diag['cloud_val_pr_auc']:.4f}")
            del data

        # ---- two-tier combination, chosen on validation ----
        self.tau_fwd, self.w = None, None
        if fog is not None and cloud is not None:
            tt = self.cfg["models"]["two_tier"]
            cal_y, cal_s = (np.concatenate([y[tr], y[va]]), np.concatenate([oof, fog[va]])) if scarce else (y[va], fog[va])
            self.tau_fwd = recall_threshold(cal_y, cal_s, tt["forward_recall"])
            # upper bound used only when the fog node is overloaded / the link is congested (bench)
            self.tau_max = max(self.tau_fwd, recall_threshold(cal_y, cal_s, tt["min_recall_under_load"]))
            best = -1
            if scarce:   # calibration set = out-of-fold training scores + validation scores
                cf, cc, cy = np.concatenate([oof, fog[va]]), np.concatenate([oof_cloud, cloud[va]]), cal_y
            else:
                cf, cc, cy = fog[va], cloud[va], y[va]
            for w in tt["blend_grid"]:
                s = self._combine(cf, cc, w)
                ap = average_precision_score(cy, s)
                if ap > best:
                    best, self.w = ap, w
            val_score = self._combine(fog[va], cloud[va], self.w)
            log.info(f"  two-tier: forward threshold {self.tau_fwd:.4f}, blend w={self.w}, val PR-AUC {best:.4f}")
            self.diag.update(blend_w=float(self.w), tau_fwd=float(self.tau_fwd))   # PATCH_REVIEW_W
        else:
            val_score = (fog if fog is not None else cloud)[va]
        self.diag["final_val_pr_auc"] = float(average_precision_score(y[va], val_score))

        # ---- decision threshold (context-adaptive unless ablated) ----
        if self.v == "fixed_threshold":
            self.thr = best_f1_threshold(y[va], val_score)
        else:
            self.thr = ContextThreshold(self.cfg["models"]["adaptive_threshold"]["min_val_pos"]).fit(
                y[va], val_score, df["role_group"].values[va], df["ctx_offhours"].values[va])
        self.train_time = time.perf_counter() - t0
        return self

    def _combine(self, fog, cloud, w):
        fwd = fog >= self.tau_fwd
        if getattr(self, "v3", False):   # PATCH_V3_FUSION: cloud confirms or dismisses; no +1 offset
            return np.where(fwd, w * cloud + (1 - w) * fog, fog)
        return np.where(fwd, 1.0 + w * cloud + (1 - w) * fog, fog)

    # ---------------------------------------------------------------- inference (any release)
    def predict(self, df: pd.DataFrame, rows: np.ndarray) -> dict:
        t0 = time.perf_counter()
        X = df[self.feats].values.astype("float32")
        Xn = self.scaler.transform(X)
        fog = cloud = None
        if self.use_anom:
            anom = self._anomaly_apply(Xn[rows])
        if self.unsup_only:
            fog = self.norm(anom[:, 0 if self.v == "if_only" else 1])
        elif self.use_fog:
            Xf = np.hstack([X[rows], anom]) if self.use_anom else X[rows]
            fog = self.lgbm.score(Xf)
        fwd_rate = 1.0
        if self.use_cloud:
            data = self._seqdata(df, Xn)
            if fog is not None:
                fwd = fog >= self.tau_fwd
                fwd_rate = float(fwd.mean())
                cloud = np.zeros(len(rows), dtype="float32")
                cloud[fwd] = self.gru.score(data, rows[fwd])     # cloud only sees forwarded events
                score = self._combine(fog, cloud, self.w)
            else:
                score = self.gru.score(data, rows)
            del data
        else:
            score, fwd_rate = fog, 0.0
        if isinstance(self.thr, float):
            pred = (score >= self.thr).astype(int)
        else:
            pred = self.thr.predict(score, df["role_group"].values[rows], df["ctx_offhours"].values[rows])
        dt = time.perf_counter() - t0
        return {"score": score, "pred": pred, "forward_rate": fwd_rate, "infer_ms_per_1k": 1e6 * dt / max(len(rows), 1)}

    # ---------------------------------------------------------------- export for fog nodes / cloud / SHAP
    def export(self, out: Path) -> None:
        """Fog bundle uses plain arrays only (loadable without torch); cloud GRU saved separately."""
        import torch
        out.mkdir(parents=True, exist_ok=True)
        self.iforest.m.set_params(n_jobs=1)   # a fog node has one core: no thread-pool overhead
        self.lgbm.booster.save_model(str(out / "fog_lgbm.txt"), num_iteration=self.lgbm.booster.best_iteration or None)
        thr = ({"global": self.thr} if isinstance(self.thr, float)
               else {"global": self.thr.glob, "context": {f"{r}|{o}": t for (r, o), t in self.thr.t.items()}})
        save_pickle({"feats": self.feats, "fog_names": self.fog_names,
                     "scaler": {"mu": self.scaler.mu, "sd": self.scaler.sd, "clip": self.scaler.clip},
                     "iforest": self.iforest.m, "ae_w": self.ae.w, "tau_fwd": self.tau_fwd,
                     "tau_max": self.tau_max, "w": self.w, "thr": thr,
                     "rf": getattr(self.lgbm, "rf", None), "rf_a": getattr(self.lgbm, "a", 0.0)},   # PATCH_V3_EXPORT
                    out / "fog_bundle.pkl")
        torch.save({"state": self.gru.net.state_dict(), "d": len(self.feats), "w": self.w,
                    "g_dim": self.gru.net.g_dim, "tau_fwd": self.tau_fwd}, out / "cloud_gru.pt")


# ============================================================================ base paper [11] reproduction
class Base11Ensemble:
    """Alkharabsheh et al. (2026): IF + PCA reconstruction + AE + LSTM-AE (7-day), min-max scores,
    weighted ensemble with weights grid-searched on validation AUC. Unsupervised training on normals."""

    def __init__(self, cfg, seed, device, ckpt_dir=None):
        self.cfg, self.seed, self.dev, self.ckpt = cfg, seed, device, ckpt_dir

    def fit(self, df, sp):
        t0 = time.perf_counter()
        self.feats = select_features(df, context=False)
        X = df[self.feats].values.astype("float32")
        y = df["label"].values.astype(int)
        tr, va = sp["train"], sp["val"]
        self.scaler = NeuralScaler(self.cfg["models"]["gru"]["clip"]).fit(X[tr])
        Xn = self.scaler.transform(X)
        trn, van = tr[y[tr] == 0], va[y[va] == 0]
        self.ifo = IForest(self.cfg, self.seed).fit(Xn[trn])
        self.pca = PCA(n_components=self.cfg["models"]["base11"]["pca_var"], random_state=self.seed).fit(
            Xn[trn][: 300000])
        ck = (lambda n: self.ckpt / n) if self.ckpt else (lambda n: None)
        self.ae = AutoEncoder(self.cfg, self.seed, self.dev, "base11").fit(Xn[trn], Xn[van], ck("b11_ae.pt"))
        data = SeqData(Xn, df["user"].values, self.cfg["models"]["base11"]["lstm_window"], self.dev)
        self.lstm = LSTMAE(self.cfg, self.seed, self.dev).fit(data, trn, van, ck("b11_lstmae.pt"))
        S_tr = self._raw(Xn, data, trn)
        self.norms = [_minmax(S_tr[:, j]) for j in range(4)]
        S_va = self._norm(self._raw(Xn, data, va))
        step = self.cfg["models"]["base11"]["weight_step"]
        grid = np.round(np.arange(0, 1 + 1e-9, step), 6)
        best = -1
        for w in itertools.product(grid, repeat=4):
            if abs(sum(w) - 1) > 1e-6:
                continue
            auc = roc_auc_score(y[va], S_va @ np.array(w))
            if auc > best:
                best, self.w = auc, np.array(w)
        self.thr = best_f1_threshold(y[va], S_va @ self.w)
        log.info(f"  [base11] weights IF/PCA/AE/LSTM-AE = {self.w.tolist()} | val AUC {best:.4f}")
        self.train_time = time.perf_counter() - t0
        return self

    def _raw(self, Xn, data, rows):
        x = Xn[rows]
        pca_err = ((self.pca.inverse_transform(self.pca.transform(x)) - x) ** 2).mean(1)
        return np.column_stack([self.ifo.score(x), pca_err, self.ae.score(x), self.lstm.score(data, rows)])

    def _norm(self, S):
        return np.column_stack([f(S[:, j]) for j, f in enumerate(self.norms)])

    def predict(self, df, rows):
        t0 = time.perf_counter()
        Xn = self.scaler.transform(df[self.feats].values.astype("float32"))
        data = SeqData(Xn, df["user"].values, self.cfg["models"]["base11"]["lstm_window"], self.dev)
        s = self._norm(self._raw(Xn, data, rows)) @ self.w
        return {"score": s, "pred": (s >= self.thr).astype(int), "forward_rate": None,
                "infer_ms_per_1k": 1e6 * (time.perf_counter() - t0) / max(len(rows), 1)}


# ============================================================================ base paper [13] reproduction
class Base13RF:
    """Alriyami & Mohamad (2026): Random Forest, 300 trees, balanced_subsample, unscaled features."""

    def __init__(self, cfg, seed, device=None, ckpt_dir=None):
        self.cfg, self.seed = cfg, seed

    def fit(self, df, sp):
        t0 = time.perf_counter()
        self.feats = select_features(df, context=getattr(self, "use_ctx", False))   # PATCH_REVIEW_RFCTX
        X = df[self.feats].values.astype("float32")
        y = df["label"].values.astype(int)
        c = self.cfg["models"]["base13_rf"]
        self.m = RandomForestClassifier(n_estimators=c["n_estimators"], class_weight=c["class_weight"],
                                        n_jobs=-1, random_state=self.seed)
        self.m.fit(X[sp["train"]], y[sp["train"]])
        self.thr = best_f1_threshold(y[sp["val"]], self.m.predict_proba(X[sp["val"]])[:, 1])
        self.train_time = time.perf_counter() - t0
        return self

    def predict(self, df, rows):
        t0 = time.perf_counter()
        s = self.m.predict_proba(df[self.feats].values[rows].astype("float32"))[:, 1]
        return {"score": s, "pred": (s >= self.thr).astype(int), "forward_rate": None,
                "infer_ms_per_1k": 1e6 * (time.perf_counter() - t0) / max(len(rows), 1)}


class Base13RFCtx(Base13RF):   # PATCH_REVIEW_RFCTX2
    """RF of [13] that also receives the context features of ConFog."""
    use_ctx = True


class ProposedV3(ProposedFramework):   # PATCH_V3_CLASS
    """ConFog v3: fog = LightGBM + shallow RF (weight on validation); rank-preserving fog-cloud fusion."""
    v3 = True


METHODS = {"ours_v3": ProposedV3, "ours": ProposedFramework, "base11": Base11Ensemble, "base13_rf": Base13RF,
           "base13_rf_ctx": Base13RFCtx}
