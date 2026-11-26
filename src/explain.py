"""Explainability (Objective 2): SHAP contributions of the fog model on the main test split.
Produces global feature importance and the alert records (with top-3 reasons) that the blockchain
stage anchors."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .common import done, load_json, log, mark_done, save_json
from .fog_runtime import FogRuntime


def run_explain(cfg: dict) -> None:
    mdir = cfg["paths"]["models"] / "fog_main"
    out = cfg["paths"]["results"] / "explain" / "alerts.jsonl"
    if done(out):
        log.info("explain: done - skip")
        return
    if not (mdir / "fog_bundle.pkl").exists():
        raise FileNotFoundError("fog_main model missing - run the experiments stage first")
    meta = load_json(mdir / "test_rows.json")
    df = pd.read_parquet(cfg["paths"]["features"] / f"{meta['release']}.parquet")
    rows = np.array(meta["test_rows"])
    fog = FogRuntime(mdir)
    X = df[fog.feats].values[rows].astype("float32")
    s = fog.score(X)
    contrib = fog.contributions(X)
    imp = pd.DataFrame({"feature": fog.names, "mean_abs_shap": np.abs(contrib).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    (cfg["paths"]["results"] / "tables").mkdir(parents=True, exist_ok=True)
    imp.to_csv(cfg["paths"]["results"] / "tables" / "shap_global.csv", index=False)
    log.info("explain: top features " + ", ".join(f"{f}={v:.3f}" for f, v in imp.head(8).values))

    # alert records = events the fog forwards to the cloud (off-chain store; hashes go on-chain)
    fwd = fog.forward(s)
    sel = np.flatnonzero(fwd)
    reasons = fog.explain(X[sel])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".tmp", "w") as f:
        for k, i in enumerate(sel):
            r = rows[i]
            f.write(json.dumps({"alert_id": f"A{k:07d}", "user": df["user"].values[r],
                                "day": str(pd.Timestamp(df["day"].values[r]).date()),
                                "fog_score": round(float(s[i]), 6), "device_id": f"FOG-{k % 3 + 1:02d}",
                                "reasons": [[n, round(v, 4)] for n, v in reasons[k]]}, sort_keys=True) + "\n")
    os.replace(str(out) + ".tmp", out)
    y = df["label"].values[rows]
    save_json({"n_test": int(len(rows)), "n_forwarded": int(fwd.sum()), "forward_rate": float(fwd.mean()),
               "recall_of_forwarding": float(fwd[y == 1].mean()) if y.sum() else None},
              cfg["paths"]["results"] / "explain" / "forwarding.json")
    mark_done(out)
    log.info(f"explain: {fwd.sum():,}/{len(rows):,} events forwarded ({fwd.mean():.2%}), "
             f"{fwd[y == 1].mean():.2%} of malicious user-days kept")
