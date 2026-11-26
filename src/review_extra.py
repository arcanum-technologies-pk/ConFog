"""Review experiments (no test-set tuning anywhere):

  tradeoff   fog model size vs accuracy and latency (RQ4). Every configuration is trained on the training
             users; the operating point is chosen on VALIDATION with a rule fixed in advance; test is
             reported once for the chosen point and for all points (Pareto curve).
  mimicry    insiders who imitate their own normal behaviour (Objective 4). The features of each malicious
             test day are moved towards that user's typical normal day by a fraction alpha; the trained
             models are NOT retrained. ConFog (full) and the re-run RF, user-disjoint split of r4.2.
  faithful   faithfulness of the three alert reasons. For every forwarded day, the three reasons (largest
             |SHAP| of the fog model) are replaced by typical values, and the score drop is compared with
             replacing three random inputs.

    python3 -m src.review_extra tradeoff mimicry faithful     -> results/review/*.json, *.csv
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .cert_features import feature_groups
from .common import get_device, load_config, load_json, log, save_json, seed_all, setup_logging
from .models import FogLGBM, NeuralScaler
from .pipeline import METHODS, ProposedFramework, select_features
from .splits import make_split

REL = "r4.2"
# selection rule fixed before any test result is seen
TOL = 0.005          # keep configurations within 0.005 validation PR-AUC of the best one
LEAVES, TREES = (7, 15, 31, 63), (50, 150, 600)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
# context the insider cannot change (role, shift, tenure, calendar)
FIXED_CTX = ("ctx_tenure", "ctx_night_shift", "ctx_shift_offset", "ctx_weekend")


def _frame(cfg):
    df = pd.read_parquet(cfg["paths"]["features"] / f"{REL}.parquet")
    df.attrs["release"] = REL
    return df


def _out(cfg):
    d = cfg["paths"]["results"] / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================================ RQ4: size / latency / accuracy
def _ae_np(w, Xn):
    h = Xn
    for i, (W, b) in enumerate(w):
        h = h @ W + b
        if i < len(w) - 1:
            h = np.maximum(h, 0)
    return ((h - Xn) ** 2).mean(1)


def _latency_ms(booster, X, scaler, iforest, ae_w, use_anom, batch=50):
    """Per-record fog latency on ONE thread, batches of 50, including the anomaly scores when used."""
    t0 = time.perf_counter()
    for i in range(0, len(X), batch):
        b = X[i:i + batch]
        if use_anom:
            Xn = scaler.transform(b)
            b = np.hstack([b, -iforest.score_samples(Xn)[:, None], _ae_np(ae_w, Xn)[:, None]])
        booster.predict(b, num_threads=1, num_iteration=booster.best_iteration or None)
    return 1000 * (time.perf_counter() - t0) / len(X)


def run_tradeoff(cfg):
    out = _out(cfg) / "tradeoff.csv"
    if out.exists():
        log.info("tradeoff: done - skip")
        return
    dev, df = get_device(cfg["device"]), _frame(cfg)
    y, users = df["label"].values.astype(int), df["user"].values
    feats = select_features(df, "full")
    X = df[feats].values.astype("float32")
    rows = []
    for proto in ("userdisj70", "chrono80"):
        for seed in cfg["seeds"][:3]:
            seed_all(seed)
            sp = make_split(df, proto, cfg, seed)
            tr, va, te = sp["train"], sp["val"], sp["test"]
            m = ProposedFramework(cfg, seed, dev, variant="fog_only")
            scaler = NeuralScaler(cfg["models"]["gru"]["clip"]).fit(X[tr])
            anom = m._anomaly(scaler.transform(X), y, users, tr, va)
            m.iforest.m.set_params(n_jobs=1)
            lat_rows = np.random.default_rng(seed).choice(te, min(5000, len(te)), replace=False)
            for use_anom in (True, False):
                Xf = np.hstack([X, anom]) if use_anom else X
                for lv in LEAVES:
                    for nt in TREES:
                        p = dict(cfg["models"]["fog_lgbm"], num_leaves=lv, n_estimators=nt)
                        f = FogLGBM(p, seed).fit(Xf[tr], y[tr], Xf[va], y[va])
                        b = f.booster
                        r = dict(protocol=proto, seed=seed, anomaly_scores=use_anom, num_leaves=lv, max_trees=nt,
                                 trees=int(b.best_iteration or b.current_iteration()),
                                 model_kb=len(b.model_to_string(num_iteration=b.best_iteration or None)) / 1024,
                                 val_pr_auc=float(average_precision_score(y[va], f.score(Xf[va]))),
                                 test_pr_auc=float(average_precision_score(y[te], f.score(Xf[te]))),
                                 ms_per_record=_latency_ms(b, X[lat_rows], scaler, m.iforest.m, m.ae.w, use_anom))
                        rows.append(r)
                        log.info(f"tradeoff {proto} s{seed} anom={use_anom} leaves={lv} trees<={nt}: "
                                 f"val {r['val_pr_auc']:.4f} test {r['test_pr_auc']:.4f} "
                                 f"{r['ms_per_record']:.4f} ms/rec {r['model_kb']:.0f} KB")
            pd.DataFrame(rows).to_csv(_out(cfg) / "tradeoff_partial.csv", index=False)
    t = pd.DataFrame(rows)
    keys = ["protocol", "anomaly_scores", "num_leaves", "max_trees"]
    agg = t.groupby(keys).agg(val=("val_pr_auc", "mean"), test=("test_pr_auc", "mean"),
                             test_sd=("test_pr_auc", "std"), ms=("ms_per_record", "mean"),
                             kb=("model_kb", "mean"), trees=("trees", "mean")).reset_index()
    choice = {}
    for proto, g in agg.groupby("protocol"):
        ok = g[g.val >= g.val.max() - TOL]                      # validation only
        pick = ok.sort_values(["ms", "kb"]).iloc[0]
        dm = g[(g.anomaly_scores) & (g.num_leaves == cfg["models"]["fog_lgbm"]["num_leaves"])
               & (g.max_trees == cfg["models"]["fog_lgbm"]["n_estimators"])]
        default = dm.iloc[0].to_dict() if len(dm) else None
        choice[proto] = {"rule": f"fastest configuration within {TOL} of the best validation PR-AUC",
                         "chosen": pick.to_dict(), "paper_default": default}
        log.info(f"tradeoff {proto}: chosen leaves={pick.num_leaves} trees<={pick.max_trees} anom={pick.anomaly_scores} "
                 f"| val {pick.val:.4f} test {pick.test:.4f} {pick.ms:.4f} ms"
                 + (f" | default test {default['test']:.4f} {default['ms']:.4f} ms" if default else ""))
    agg.to_csv(_out(cfg) / "tradeoff_mean.csv", index=False)
    save_json(choice, _out(cfg) / "tradeoff_choice.json")
    t.to_csv(out, index=False)


# ============================================================================ Objective 4: mimicry insiders
def _mimic(df, rows_mal, cols, alpha, prof):
    d = df.copy()
    d.attrs["release"] = REL
    if alpha == 0:
        return d
    users = d["user"].values[rows_mal]
    P = prof.loc[users, cols].values.astype("float64")
    V = d[cols].values[rows_mal].astype("float64")
    binary = np.array([bool(d[c].isin([0, 1]).all()) for c in cols])
    new = V + alpha * (P - V)
    if binary.any():                      # yes/no inputs (e.g. off-hours) switch at alpha >= 0.5
        new[:, binary] = P[:, binary] if alpha >= 0.5 else V[:, binary]
    A = d[cols].to_numpy(dtype="float64", copy=True)
    A[rows_mal] = new
    d[cols] = pd.DataFrame(A, index=d.index, columns=cols).astype({c: "float32" for c in cols})
    return d


def run_mimicry(cfg):
    out = _out(cfg) / "mimicry.csv"
    if out.exists():
        log.info("mimicry: done - skip")
        return
    dev, df = get_device(cfg["device"]), _frame(cfg)
    y = df["label"].values.astype(int)
    g = feature_groups(df)
    cols = g["behaviour"] + g["deviation"] + [c for c in g["context"] if c not in FIXED_CTX
                                               and not c.startswith("ctx_role")]
    cols = [c for c in dict.fromkeys(cols) if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for seed in cfg["seeds"]:
        seed_all(seed)
        sp = make_split(df, "userdisj70", cfg, seed)
        te = sp["test"]
        mal = te[y[te] == 1]
        # typical normal day of each test user (median over that user's normal days)
        norm = df.iloc[te[y[te] == 0]]
        prof = norm.groupby("user")[cols].median()
        mal = mal[np.isin(df["user"].values[mal], prof.index)]
        models = {"ConFog": METHODS["ours"](cfg, seed, dev, variant="full", ckpt_dir=None).fit(df, sp),
                  "RF (re-run)": METHODS["base13_rf"](cfg, seed, dev, ckpt_dir=None).fit(df, sp)}
        for a in ALPHAS:
            d = _mimic(df, mal, cols, a, prof)
            for name, m in models.items():
                o = m.predict(d, te)
                yt = y[te]
                ismal = np.isin(te, mal)
                r = dict(seed=seed, model=name, alpha=a, n_malicious=int(ismal.sum()),
                         recall=float(o["pred"][ismal].mean()),
                         pr_auc=float(average_precision_score(yt, o["score"])),
                         forwarded_malicious=(float((o["score"][ismal] >= 1.0).mean()) if name == "ConFog" else None))
                rows.append(r)
                log.info(f"mimicry s{seed} {name:12s} alpha={a:.2f}: recall {r['recall']:.3f} PR-AUC {r['pr_auc']:.4f}")
            del d
        pd.DataFrame(rows).to_csv(_out(cfg) / "mimicry_partial.csv", index=False)
    t = pd.DataFrame(rows)
    t.groupby(["model", "alpha"]).agg(recall=("recall", "mean"), recall_sd=("recall", "std"),
                                      pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std")).reset_index() \
        .to_csv(_out(cfg) / "mimicry_mean.csv", index=False)
    t.to_csv(out, index=False)


# ============================================================================ explainability: faithfulness
def run_faithful(cfg, k_list=(1, 3, 5), n_random=20):
    from scipy.stats import wilcoxon

    from .fog_runtime import FogRuntime
    out = _out(cfg) / "faithfulness.json"
    if out.exists():
        log.info("faithful: done - skip")
        return
    mdir = cfg["paths"]["models"] / "fog_main"
    meta = load_json(mdir / "test_rows.json")
    df = pd.read_parquet(cfg["paths"]["features"] / f"{meta['release']}.parquet")
    rows = np.array(meta["test_rows"])
    fog = FogRuntime(mdir)
    inp = fog.inputs(df[fog.feats].values[rows].astype("float32"))
    y = df["label"].values[rows].astype(int)
    b = fog.booster
    s = b.predict(inp)
    sel = np.flatnonzero(s >= fog.tau_fwd)                      # days that carry reasons (forwarded)
    base = np.median(inp[y == 0], axis=0)                       # typical value of every input
    C = b.predict(inp[sel], pred_contrib=True)[:, :-1]
    order = np.argsort(-np.abs(C), axis=1)                      # same ranking as the alert reasons
    rng = np.random.default_rng(0)
    res = {"n_alerts": int(len(sel)), "n_malicious_alerts": int(y[sel].sum()), "baseline": "median of normal test days"}
    for k in k_list:
        Xt = inp[sel].copy()
        idx = order[:, :k]
        Xt[np.arange(len(sel))[:, None], idx] = base[idx]
        d_top = s[sel] - b.predict(Xt)
        d_rnd = np.zeros(len(sel))
        for _ in range(n_random):
            Xr = inp[sel].copy()
            ridx = np.array([rng.choice(inp.shape[1], k, replace=False) for _ in range(len(sel))])
            Xr[np.arange(len(sel))[:, None], ridx] = base[ridx]
            d_rnd += (s[sel] - b.predict(Xr)) / n_random
        stat = wilcoxon(d_top, d_rnd, alternative="greater") if len(sel) > 10 else None
        res[f"top{k}"] = {"mean_drop_reasons": float(d_top.mean()), "mean_drop_random": float(d_rnd.mean()),
                          "share_reasons_larger": float((d_top > d_rnd).mean()),
                          "share_no_longer_forwarded_reasons": float(((s[sel] - d_top) < fog.tau_fwd).mean()),
                          "share_no_longer_forwarded_random": float(((s[sel] - d_rnd) < fog.tau_fwd).mean()),
                          "wilcoxon_p": float(stat.pvalue) if stat is not None else None}
        log.info(f"faithful top{k}: drop {d_top.mean():.3f} vs random {d_rnd.mean():.3f} | "
                 f"larger in {(d_top > d_rnd).mean():.1%} of {len(sel)} alerts")
    save_json(res, out)


if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["paths"]["logs"], "review_extra")
    for job in sys.argv[1:] or ["tradeoff", "mimicry", "faithful"]:
        {"tradeoff": run_tradeoff, "mimicry": run_mimicry, "faithful": run_faithful}[job](cfg)
