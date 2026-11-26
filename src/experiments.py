"""Experiment runner. Every (protocol, dataset, method, variant, seed) is one task whose result is
saved atomically; finished tasks are skipped on restart, unfinished neural training resumes from its
last epoch checkpoint."""
from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import fmt_s, get_device, log, save_json, seed_all
from .iot_data import OrdinalEncoder, is_device_insider, load_ton
from .metrics import binary_report, multiclass_report
from .models import balanced_weights, train_lgbm
from .pipeline import METHODS
from .splits import make_split


# ============================================================================ CERT task list
def cert_tasks(cfg: dict) -> list[dict]:
    e, seeds, T = cfg["experiments"], cfg["seeds"], []
    for s in seeds:
        for rel in e["chrono80"]:
            for m in ("ours", "base11", "base13_rf"):
                T.append(dict(protocol="chrono80", release=rel, method=m, variant="full", seed=s))
        for rel in e["chrono70"]:
            for m in ("ours", "base13_rf"):
                T.append(dict(protocol="chrono70", release=rel, method=m, variant="full", seed=s))
        for rel in e["userdisj70"]:
            for m in ("ours", "base13_rf"):
                T.append(dict(protocol="userdisj70", release=rel, method=m, variant="full", seed=s))
        for rel in e["loso"]:
            for m in ("ours", "base11", "base13_rf"):
                T.append(dict(protocol="loso", release=rel, method=m, variant="full", seed=s, scenario=None))
        for v in e["ablations"]:
            T.append(dict(protocol=e["ablation_protocol"], release=e["ablation_release"], method="ours",
                          variant=v, seed=s))
        # PATCH_V3_TASKS: ConFog v3 on every protocol
        for proto, rels in (("chrono80", ["r4.2"]), ("chrono70", ["r4.2", "r5.2"]),
                            ("userdisj70", ["r4.2", "r5.2", "r6.2"]), ("loso", ["r4.2", "r5.2"])):
            for rel in rels:
                for v in (["full"] if rel == "r6.2" else ["full", "fog_only", "no_graph", "fixed_threshold"]):
                    T.append(dict(protocol=proto, release=rel, method="ours_v3", variant=v, seed=s,
                                  **({"scenario": None} if proto == "loso" else {})))
        # PATCH_DEEP_TASKS: deep sequence baseline (cloud model alone) on the other protocols
        for proto, rels in (("userdisj70", ["r4.2", "r5.2"]), ("loso", ["r4.2", "r5.2"]), ("chrono70", ["r4.2", "r5.2"])):
            for rel in rels:
                T.append(dict(protocol=proto, release=rel, method="ours", variant="cloud_only", seed=s,
                              **({"scenario": None} if proto == "loso" else {})))
        # PATCH_REVIEW_TASKS
        for proto, rels, variants in (("userdisj70", ["r4.2", "r5.2"],
                                       ["fog_only", "no_graph", "fixed_threshold", "no_context", "no_tenure", "no_web_keywords"]),
                                      ("loso", ["r4.2", "r5.2"], ["fog_only", "no_graph", "fixed_threshold"]),
                                      ("chrono80", ["r4.2"], ["no_tenure", "no_web_keywords"])):
            for rel in rels:
                for v in variants:
                    T.append(dict(protocol=proto, release=rel, method="ours", variant=v, seed=s,
                                  **({"scenario": None} if proto == "loso" else {})))
        for proto, rels in (("chrono80", ["r4.2"]), ("chrono70", ["r4.2", "r5.2"]), ("userdisj70", ["r4.2", "r5.2"])):
            for rel in rels:
                T.append(dict(protocol=proto, release=rel, method="base13_rf_ctx", variant="full", seed=s))
    uniq, seen = [], set()
    for t in T:
        k = task_name(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    order = {r: i for i, r in enumerate(cfg["cert"]["releases"])}
    return sorted(uniq, key=lambda t: (order.get(t["release"], 9), t["seed"]))


def task_name(t: dict) -> str:
    sc = f"/scen{t['scenario']}" if t.get("scenario") else ""
    return f"{t['protocol']}/{t['release']}/{t['method']}/{t['variant']}{sc}/seed{t['seed']}"


class Frames:
    """Loads release feature tables on demand and keeps at most `keep` in memory."""

    def __init__(self, cfg, keep=2):
        self.cfg, self.keep, self.cache = cfg, keep, {}

    def __getitem__(self, rel):
        if rel not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            df = pd.read_parquet(self.cfg["paths"]["features"] / f"{rel}.parquet")
            df.attrs["release"] = rel
            self.cache[rel] = df
        return self.cache[rel]


def _scenarios(df):
    return sorted(int(s) for s in df["scenario"].unique() if s > 0)


def run_cert(cfg: dict) -> None:
    dev = get_device(cfg["device"])
    raw = cfg["paths"]["results"] / "raw"
    frames = Frames(cfg)
    tasks = []
    for t in cert_tasks(cfg):   # expand LOSO into one task per scenario
        if t["protocol"] == "loso":
            for sc in _scenarios(frames[t["release"]]):
                tasks.append({**t, "scenario": sc})
        else:
            tasks.append(t)
    todo = [t for t in tasks if not (raw / f"{task_name(t)}.json").exists()]
    log.info(f"CERT experiments: {len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run "
             f"(device={dev})")
    failed, t_all = [], time.time()
    for i, t in enumerate(todo, 1):
        name = task_name(t)
        log.info(f"[task {i}/{len(todo)}] {name}")
        try:
            _run_one(cfg, t, frames, dev, raw / f"{name}.json")
        except Exception as ex:  # one failing task must not stop the whole run
            failed.append(name)
            log.error(f"[task {i}/{len(todo)}] FAILED {name}: {ex}\n{traceback.format_exc()}")
        el = time.time() - t_all
        log.info(f"[progress] {i}/{len(todo)} | elapsed {fmt_s(el)} | ETA {fmt_s(el / i * (len(todo) - i))}")
    if failed:
        log.warning(f"{len(failed)} tasks failed (re-run to retry): {failed}")


def _run_one(cfg, t, frames, dev, out):
    seed_all(t["seed"])
    df = frames[t["release"]]
    source_only = t["protocol"] == "userdisj70" and t["release"] not in cfg["experiments"]["cross_targets"]
    sp = make_split(df, t["protocol"], cfg, t["seed"], t.get("scenario"), source_only)
    ck = cfg["paths"]["models"] / "ckpt" / task_name(t)
    Model = METHODS[t["method"]]
    model = (Model(cfg, t["seed"], dev, variant=t["variant"], ckpt_dir=ck) if t["method"].startswith("ours")   # PATCH_V3_RUNNER
             else Model(cfg, t["seed"], dev, ckpt_dir=ck)).fit(df, sp)

    evals = [(t["release"], df, sp["test"])] if len(sp["test"]) else []
    if t["protocol"] == "userdisj70":   # zero-shot cross-dataset transfer (protocol of [13])
        for tgt in cfg["experiments"]["cross_targets"]:
            if tgt != t["release"]:
                dft = frames[tgt]
                evals.append((tgt, dft, make_split(dft, "userdisj70", cfg, t["seed"])["test"]))
    rows = []
    for tgt, d, r in evals:
        o = model.predict(d, r)
        y = d["label"].values[r].astype(int)
        m = binary_report(y, o["score"], o["pred"])
        m.update(task=task_name(t), **{k: v for k, v in t.items()}, target=tgt,
                 forward_rate=o["forward_rate"], infer_ms_per_1k=o["infer_ms_per_1k"],
                 train_time_s=model.train_time, **getattr(model, "diag", {}))
        rows.append(m)
        log.info(f"  -> test on {tgt}: PR-AUC {m['pr_auc']:.4f} ROC-AUC {m['roc_auc']:.4f} "
                 f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} FPR {m['fpr']:.4f}"
                 + (f" fwd {o['forward_rate']:.3f}" if o["forward_rate"] is not None else ""))
        if tgt == t["release"]:   # kept for PR curves / significance tests
            sdir = cfg["paths"]["results"] / "scores"
            sdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sdir / f"{task_name(t).replace('/', '__')}.npz", y=y, score=o["score"])
    main = (t["protocol"] == cfg["experiments"]["ablation_protocol"] and t["release"] == cfg["experiments"]["ablation_release"]
            and t["method"] == "ours" and t["variant"] == "full" and t["seed"] == cfg["seeds"][0])
    if main:
        model.export(cfg["paths"]["models"] / "fog_main")
        save_json({"protocol": t["protocol"], "release": t["release"], "test_rows": sp["test"].tolist()},
                  cfg["paths"]["models"] / "fog_main" / "test_rows.json")
    save_json(rows, out)


# ============================================================================ IoT (fog-layer device models)
def _fit_eval(cfg, Xtr, ytr, Xva, yva, Xte, yte, n_class, seed) -> list[dict]:
    """Fog (small) and cloud (large) LightGBM device models + XGBoost as used in base paper [15]."""
    from xgboost import XGBClassifier
    out = []
    for name in ("fog_lgbm", "cloud_lgbm", "xgb_base15"):
        t0 = time.perf_counter()
        if name == "xgb_base15":
            m = XGBClassifier(**cfg["iot"]["xgb_base15"], random_state=seed, n_jobs=-1, tree_method="hist")
            m.fit(Xtr, ytr)
            predict = m.predict_proba
        else:
            b = train_lgbm(cfg["iot"][name], Xtr, ytr, Xva, yva, n_class, seed, weight=balanced_weights(ytr))
            it = b.best_iteration or None
            predict = (lambda X, b=b, it=it: (lambda p: np.column_stack([1 - p, p]) if p.ndim == 1 else p)(
                b.predict(X, num_iteration=it)))
        tt = time.perf_counter() - t0
        t1 = time.perf_counter()
        proba = predict(Xte)
        it_s = time.perf_counter() - t1
        r = multiclass_report(yte, proba.argmax(1), proba[:, 1] if n_class == 2 else None)
        r.update(model=name, train_time_s=tt, infer_ms_per_1k=1e6 * it_s / max(len(yte), 1))
        out.append(r)
    return out


def run_iot(cfg: dict) -> None:
    raw = cfg["paths"]["results"] / "raw" / "iot"
    for fn in (_ton, _ciciomt):
        try:
            fn(cfg, raw)
        except FileNotFoundError as ex:
            log.warning(f"IoT dataset skipped: {ex}")


def _ton(cfg, raw):
    c = cfg["iot"]["ton_iot"]
    X = y = None
    for seed in cfg["seeds"]:
        for proto in ("cv5", "holdout"):
            out = raw / f"ton_iot/{proto}/seed{seed}.json"
            if out.exists():
                continue
            if X is None:
                X, y, cat = load_ton(cfg)
            seed_all(seed)
            rows = []
            if proto == "cv5":   # same size and validation scheme as base paper [15]
                idx, _ = train_test_split(np.arange(len(y)), train_size=min(c["sample_n"], len(y) - 2),
                                          stratify=y, random_state=seed)
                Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
                for k, (a, b) in enumerate(StratifiedKFold(c["cv_folds"], shuffle=True, random_state=seed).split(Xs, ys)):
                    a, v = train_test_split(a, test_size=0.15, stratify=ys[a], random_state=seed)
                    enc = OrdinalEncoder().fit(Xs.iloc[a], cat)
                    T = [enc.transform(Xs.iloc[j]).values for j in (a, v, b)]
                    for r in _fit_eval(cfg, T[0], ys[a], T[1], ys[v], T[2], ys[b], 2, seed):
                        rows.append({**r, "fold": k})
            else:
                f_tr, f_va, _ = c["holdout"]
                a, rest = train_test_split(np.arange(len(y)), train_size=f_tr, stratify=y, random_state=seed)
                v, b = train_test_split(rest, train_size=f_va / (1 - f_tr), stratify=y[rest], random_state=seed)
                enc = OrdinalEncoder().fit(X.iloc[a], cat)
                T = [enc.transform(X.iloc[j]).values for j in (a, v, b)]
                rows = _fit_eval(cfg, T[0], y[a], T[1], y[v], T[2], y[b], 2, seed)
            for r in rows:
                r.update(dataset="ton_iot", protocol=proto, seed=seed)
            save_json(rows, out)
            log.info(f"[ToN-IoT] {proto} seed{seed}: " + ", ".join(
                f"{r['model']} acc={r['accuracy']:.4f} F1={r['f1_macro']:.4f}" for r in rows[-3:]))


def _ciciomt(cfg, raw):
    from .iot_data import CATEGORIES, prepare_ciciomt
    p_tr, p_te = prepare_ciciomt(cfg)
    tr0 = te0 = None
    for seed in cfg["seeds"]:
        for task in ("binary", "category", "device_insider"):
            out = raw / f"ciciomt2024/{task}/seed{seed}.json"
            if out.exists():
                continue
            if tr0 is None:
                tr0, te0 = pd.read_parquet(p_tr), pd.read_parquet(p_te)
            seed_all(seed)
            tr, te = tr0, te0
            if task == "device_insider":
                keep = lambda d: (d["category"] == "Benign") | d["label_name"].map(is_device_insider)
                tr, te = tr.loc[keep(tr)], te.loc[keep(te)]
            if task == "category":
                present = [c for c in CATEGORIES if c in set(tr["category"])]   # classes present in train
                te = te[te["category"].isin(present)]
                cmap = {c: i for i, c in enumerate(present)}
                ytr, yte = tr["category"].map(cmap).values, te["category"].map(cmap).values
                n_class = len(present)
            else:
                ytr, yte = (tr["category"] != "Benign").astype(int).values, (te["category"] != "Benign").astype(int).values
                n_class = 2
            feats = [x for x in tr.columns if x not in ("label_name", "category")]
            a, v = train_test_split(np.arange(len(ytr)), test_size=cfg["iot"]["ciciomt2024"]["val_frac"],
                                    stratify=ytr, random_state=seed)
            Xtr = tr[feats].values
            rows = _fit_eval(cfg, Xtr[a], ytr[a], Xtr[v], ytr[v], te[feats].values, yte, n_class, seed)
            for r in rows:
                r.update(dataset="ciciomt2024", protocol=task, seed=seed)
            save_json(rows, out)
            log.info(f"[CICIoMT2024] {task} seed{seed}: " + ", ".join(
                f"{r['model']} acc={r['accuracy']:.4f} F1={r['f1_macro']:.4f}" for r in rows))
