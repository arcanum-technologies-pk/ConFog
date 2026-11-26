"""Models: fog (Isolation Forest, autoencoder, LightGBM), cloud (GRU), base-paper reproductions
(LSTM autoencoder for [11]). Neural training checkpoints every epoch and resumes after a crash."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from .common import log


# ============================================================================ preprocessing
class NeuralScaler:
    """signed log1p -> standardise (train statistics) -> clip. Fitted on training rows only."""

    def __init__(self, clip: float = 8.0):
        self.clip = clip

    @staticmethod
    def _slog(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X: np.ndarray):
        z = self._slog(X.astype("float64"))
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((self._slog(X) - self.mu) / self.sd, -self.clip, self.clip).astype("float32")


def window_index(users: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by (user, day): idx[i] = the `window` most recent rows of the same user
    ending at i (left-padded), mask = 1 for valid positions. Only past/current rows are used."""
    n = len(users)
    start = np.zeros(n, dtype=np.int64)
    change = np.flatnonzero(users[1:] != users[:-1]) + 1
    start[change] = change
    start = np.maximum.accumulate(start)
    offs = np.arange(-window + 1, 1)
    idx = np.arange(n)[:, None] + offs[None, :]
    mask = idx >= start[:, None]
    idx = np.where(mask, idx, np.arange(n)[:, None])
    return idx.astype(np.int64), mask.astype(np.float32)


# ============================================================================ anomaly detectors
class IForest:
    def __init__(self, cfg: dict, seed: int):
        c = cfg["models"]["iforest"]
        self.max_train = c["max_train"]
        self.m = IsolationForest(n_estimators=c["n_estimators"], max_samples=c["max_samples"],
                                 random_state=seed, n_jobs=-1)
        self.seed = seed

    def fit(self, X):
        if len(X) > self.max_train:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)]
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class _MLPAE(nn.Module):
    def __init__(self, d: int, hidden: list[int]):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(h2, h1), nn.ReLU(), nn.Linear(h1, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoEncoder:
    """Shallow AE trained on normal rows. Inference is exported to NumPy (no torch on fog nodes)."""

    def __init__(self, cfg: dict, seed: int, device, tag: str = ""):
        self.c, self.seed, self.dev, self.tag = cfg["models"]["ae"], seed, device, tag

    def fit(self, Xtr, Xva, ckpt: Path | None = None):
        c = self.c
        rng = np.random.default_rng(self.seed)
        if len(Xtr) > c["max_train"]:
            Xtr = Xtr[rng.choice(len(Xtr), c["max_train"], replace=False)]
        net = _MLPAE(Xtr.shape[1], c["hidden"]).to(self.dev)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
        xt = torch.from_numpy(Xtr).to(self.dev)
        xv = torch.from_numpy(Xva[: 100000]).to(self.dev)

        def epoch_fn(ep):
            perm = torch.randperm(len(xt), device=self.dev)
            for i in range(0, len(xt), c["batch"]):
                b = xt[perm[i:i + c["batch"]]]
                loss = ((net(b) - b) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                return -float(((net(xv) - xv) ** 2).mean())  # higher is better
        fit_loop(net, opt, epoch_fn, c["epochs"], c["patience"], ckpt, f"AE[{self.tag}]")
        self.w = [(l.weight.detach().cpu().numpy().T, l.bias.detach().cpu().numpy())
                  for l in list(net.enc) + list(net.dec) if isinstance(l, nn.Linear)]
        return self

    def score(self, X):
        h = X
        for i, (W, b) in enumerate(self.w):
            h = h @ W + b
            if i < len(self.w) - 1:
                h = np.maximum(h, 0)
        return ((h - X) ** 2).mean(1)


# ============================================================================ training loop with resume
def fit_loop(net, opt, epoch_fn, epochs: int, patience: int, ckpt: Path | None, name: str):
    """Runs epoch_fn(ep) -> validation score (higher better); keeps best weights; early stopping;
    checkpoints model/optimizer/progress after every epoch so a crash resumes mid-training."""
    start, best, best_state, bad = 0, -np.inf, None, 0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, best, best_state, bad = st["epoch"] + 1, st["best"], st["best_state"], st["bad"]
        torch.set_rng_state(st["rng"])
        log.info(f"    {name}: resumed from epoch {start} (best={best:.4f})")
    ep = start - 1
    for ep in range(start, epochs):
        if bad >= patience:
            ep -= 1
            break
        net.train()
        score = epoch_fn(ep)
        net.eval()
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0:
            log.info(f"    {name} epoch {ep:3d} | val {score:.4f} | best {best:.4f} | patience {bad}/{patience}")
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                        "best_state": best_state, "bad": bad, "rng": torch.get_rng_state()},
                       str(ckpt) + ".tmp")
            Path(str(ckpt) + ".tmp").replace(ckpt)
    if best_state is not None:
        net.load_state_dict(best_state)
    log.info(f"    {name} done: {ep + 1} epochs, best val {best:.4f}")
    net.eval()
    return net


# ============================================================================ gradient boosting (fog / device models)
def train_lgbm(params: dict, Xtr, ytr, Xva, yva, n_class: int, seed: int, weight=None,
               feature_names=None) -> lgb.Booster:
    """LightGBM native API (stable across versions). Early stopping on the validation set only."""
    p = dict(params)
    rounds = p.pop("n_estimators")
    es = p.pop("early_stopping", None)
    p.update(seed=seed, num_threads=-1, verbose=-1, feature_pre_filter=False)
    if n_class == 2:
        p.update(objective="binary", metric="average_precision")
    else:
        p.update(objective="multiclass", num_class=n_class, metric="multi_logloss")
    dtr = lgb.Dataset(Xtr, ytr, weight=weight, feature_name=feature_names or "auto")
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    cb = [lgb.early_stopping(es, verbose=False)] if es else []
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=cb)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(cls) * n) for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y], dtype="float32")


class FogLGBM:
    def __init__(self, params: dict, seed: int):
        self.p, self.seed = params, seed

    def _params(self, y):
        """min_child_samples never exceeds ~10 % of the positive class (r6.2 has only ~35 positive days)."""
        neg, pos = (y == 0).sum(), max((y == 1).sum(), 1)
        mcs = int(np.clip(pos // 10, 5, self.p["min_child_samples"]))
        return dict(self.p, scale_pos_weight=float(np.sqrt(min(neg / pos, 1e4))), min_child_samples=mcs)

    def fit(self, Xtr, ytr, Xva, yva, feature_names=None):
        p = self._params(ytr)
        self.booster = train_lgbm(p, Xtr, ytr, Xva, yva, 2, self.seed, feature_names=feature_names)
        log.info(f"    LightGBM: {self.booster.best_iteration or self.booster.current_iteration()} trees")
        return self

    def fit_cv(self, X, y, folds, feature_names=None):
        """For a validation set with too few positives to select on: the number of trees is chosen by
        user-grouped cross-validation on the training data; returns out-of-fold scores of the training rows."""
        p = self._params(y)
        rounds, es = p.pop("n_estimators"), p.pop("early_stopping", None)
        p.update(objective="binary", metric="average_precision", seed=self.seed, num_threads=-1, verbose=-1,
                 feature_pre_filter=False)
        cv = lgb.cv(p, lgb.Dataset(X, y, feature_name=feature_names or "auto"), num_boost_round=rounds,
                    folds=folds, callbacks=[lgb.early_stopping(es, verbose=False)] if es else [],
                    return_cvbooster=True)
        best = len(next(v for k, v in cv.items() if k.endswith("-mean")))
        oof = np.zeros(len(y))
        for (_, te), b in zip(folds, cv["cvbooster"].boosters):
            oof[te] = b.predict(X[te], num_iteration=best)
        self.booster = lgb.train(p, lgb.Dataset(X, y, feature_name=feature_names or "auto"), num_boost_round=best)
        log.info(f"    LightGBM: {best} trees (chosen by grouped CV on training data)")
        return oof

    def score(self, X):
        return self.booster.predict(X, num_iteration=self.booster.best_iteration or None)


class FogBlend:   # PATCH_V3_FOGBLEND
    """ConFog v3 fog model: LightGBM averaged with a shallow random forest; weight a chosen on validation."""
    GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, params: dict, seed: int):
        from sklearn.ensemble import RandomForestClassifier
        self.lgb, self.seed, self.a = FogLGBM(params, seed), seed, 0.0
        self.rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20, max_features="sqrt",
                                         class_weight="balanced_subsample", n_jobs=-1, random_state=seed)

    @property
    def booster(self):
        return self.lgb.booster

    def _pick(self, y, l, r):
        from sklearn.metrics import average_precision_score
        self.a = float(max(self.GRID, key=lambda a: (average_precision_score(y, (1 - a) * l + a * r), -a)))
        log.info(f"    fog blend: LightGBM + RF, a={self.a}")

    def fit(self, Xtr, ytr, Xva, yva, feature_names=None):
        self.lgb.fit(Xtr, ytr, Xva, yva, feature_names)
        self.rf.fit(Xtr, ytr)
        self._pick(yva, self.lgb.score(Xva), self.rf.predict_proba(Xva)[:, 1])
        return self

    def fit_cv(self, X, y, folds, feature_names=None):
        from sklearn.base import clone
        oof_l = self.lgb.fit_cv(X, y, folds, feature_names)
        oof_r = np.zeros(len(y))
        for a_idx, b_idx in folds:
            oof_r[b_idx] = clone(self.rf).fit(X[a_idx], y[a_idx]).predict_proba(X[b_idx])[:, 1]
        self.rf.fit(X, y)
        self._pick(y, oof_l, oof_r)
        return (1 - self.a) * oof_l + self.a * oof_r

    def score(self, X):
        s = self.lgb.score(X)
        return s if self.a == 0 else (1 - self.a) * s + self.a * self.rf.predict_proba(X)[:, 1]


# ============================================================================ cloud sequence model
class _SeqGraph(nn.Module):
    """Cloud model: GRU over the user's last days  +  graph branch over neighbour aggregations
    ([x, A x, A^2 x, log degree], SIGN-style); both embeddings are fused for the final score."""

    def __init__(self, d, g_dim, hidden, dropout):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.g_dim = g_dim
        if g_dim:
            self.graph = nn.Sequential(nn.Linear(g_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * (2 if g_dim else 1), hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, g=None):
        _, h = self.gru(x)
        z = h[-1] if not self.g_dim else torch.cat([h[-1], self.graph(g)], dim=1)
        return self.head(z).squeeze(-1)


class SeqData:
    """Holds a whole release on the GPU and gathers windows on the fly (no window copies in RAM).
    G (optional) = per-user-day graph inputs for the graph branch."""

    def __init__(self, X: np.ndarray, users: np.ndarray, window: int, device, G: np.ndarray | None = None):
        idx, mask = window_index(users, window)
        self.X = torch.from_numpy(X).to(device)
        self.idx = torch.from_numpy(idx).to(device)
        self.mask = torch.from_numpy(mask).to(device)
        self.G = torch.from_numpy(G).to(device) if G is not None else None

    def batch(self, rows: torch.Tensor) -> torch.Tensor:
        return self.X[self.idx[rows]] * self.mask[rows].unsqueeze(-1)

    def graph(self, rows: torch.Tensor):
        return None if self.G is None else self.G[rows]


@torch.no_grad()
def _predict_seq(net, data: SeqData, rows: np.ndarray, bs: int = 8192) -> np.ndarray:
    out = []
    r = torch.from_numpy(rows).to(data.X.device)
    for i in range(0, len(r), bs):
        b = r[i:i + bs]
        out.append(torch.sigmoid(net(data.batch(b), data.graph(b))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


class CloudGRU:
    def __init__(self, cfg: dict, seed: int, device):
        self.c, self.seed, self.dev = cfg["models"]["gru"], seed, device

    def fit(self, data: SeqData, y: np.ndarray, tr: np.ndarray, va: np.ndarray, ckpt: Path | None = None,
            select_on_val: bool = True):
        """select_on_val=False (validation has too few positives): train a fixed number of epochs
        (models.gru.fixed_epochs) and keep the last weights instead of early stopping on a noisy score."""
        c = dict(self.c)
        if not select_on_val:
            c.update(epochs=c["fixed_epochs"], patience=c["fixed_epochs"])
        g_dim = 0 if data.G is None else data.G.shape[1]
        net = _SeqGraph(data.X.shape[1], g_dim, c["hidden"], c["dropout"]).to(self.dev)
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        # every epoch sees >= min_epoch_samples rows (positives re-sampled when they are few), so a release
        # with ~35 positive days still gets a real number of gradient steps
        n_pos = max(len(pos), int(np.ceil(c["min_epoch_samples"] / (1 + c["neg_ratio"]))))
        n_neg = min(len(neg), c["neg_ratio"] * n_pos)
        lossf = nn.BCEWithLogitsLoss()
        yt = torch.from_numpy(y.astype("float32")).to(self.dev)

        def epoch_fn(ep):
            g = np.random.default_rng(self.seed * 1000 + ep)   # fresh negatives every epoch
            rows = np.concatenate([g.choice(pos, n_pos, replace=n_pos > len(pos)), g.choice(neg, n_neg, replace=False)])
            rows = torch.from_numpy(rows[g.permutation(len(rows))]).to(self.dev)
            for i in range(0, len(rows), c["batch"]):
                b = rows[i:i + c["batch"]]
                loss = lossf(net(data.batch(b), data.graph(b)), yt[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            if not select_on_val:
                return float(ep)                        # monotone -> the last epoch is kept
            s = _predict_seq(net, data, va)
            net.train()
            return float(average_precision_score(y[va], s))
        self.net = fit_loop(net, opt, epoch_fn, c["epochs"], c["patience"], ckpt, ("GRU+graph" if g_dim else "GRU") + ("" if select_on_val else " (fixed epochs; value = epoch)"))
        return self

    def score(self, data: SeqData, rows: np.ndarray) -> np.ndarray:
        return _predict_seq(self.net, data, rows)


# ============================================================================ [11] LSTM autoencoder
class _LSTMAE(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.enc = nn.LSTM(d, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


class LSTMAE:
    """LSTM autoencoder over 7-day windows, trained on normal user-days (as in base paper [11])."""

    def __init__(self, cfg: dict, seed: int, device):
        self.c, self.b, self.seed, self.dev = cfg["models"]["ae"], cfg["models"]["base11"], seed, device

    def fit(self, data: SeqData, tr_norm: np.ndarray, va_norm: np.ndarray, ckpt: Path | None = None):
        c = self.c
        net = _LSTMAE(data.X.shape[1], self.b["lstm_hidden"]).to(self.dev)
        opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
        g = np.random.default_rng(self.seed)
        tr = tr_norm if len(tr_norm) <= c["max_train"] else g.choice(tr_norm, c["max_train"], replace=False)
        va = torch.from_numpy(va_norm[:50000]).to(self.dev)
        tr_t = torch.from_numpy(tr).to(self.dev)

        def epoch_fn(ep):
            perm = tr_t[torch.randperm(len(tr_t), device=self.dev)]
            for i in range(0, len(perm), c["batch"]):
                x = data.batch(perm[i:i + c["batch"]])
                loss = ((net(x) - x) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                x = data.batch(va)
                return -float(((net(x) - x) ** 2).mean())
        self.net = fit_loop(net, opt, epoch_fn, c["epochs"], c["patience"], ckpt, "LSTM-AE")
        return self

    @torch.no_grad()
    def score(self, data: SeqData, rows: np.ndarray, bs: int = 8192) -> np.ndarray:
        out, r = [], torch.from_numpy(rows).to(self.dev)
        for i in range(0, len(r), bs):
            x = data.batch(r[i:i + bs])
            out.append(((self.net(x) - x) ** 2).mean((1, 2)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
