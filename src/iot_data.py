"""Device-side datasets (fog layer, compromised trusted devices).

ToN-IoT (network)  - comparison with base paper [15].
CICIoMT2024        - real medical-device traffic (novelty of this work).

Leakage controls: identifier/time columns dropped, exact duplicates removed before splitting,
test rows identical to a training row removed, encoders fitted on training data only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, done, log, mark_done, save_parquet

CATEGORIES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]


# ============================================================================ ToN-IoT
def load_ton(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """ToN-IoT network data: the UNSW train_test_network.csv or the CIC-ToN-IoT version
    (CICFlowMeter features, Kaggle dhoogla/cictoniot, CSV or parquet) - whatever is in data/raw/ton_iot/."""
    c = cfg["iot"]["ton_iot"]
    folder = cfg["paths"]["raw"] / "ton_iot"
    files = sorted(folder.glob("*.parquet")) + sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV/parquet file in {folder} - see README (ToN-IoT download)")
    df = pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
                    for f in files], ignore_index=True)
    df.columns = [x.strip() for x in df.columns]
    lab = next(x for x in df.columns if x.lower() == "label")
    y = df[lab]
    if not pd.api.types.is_numeric_dtype(y):                       # text labels -> attack / benign
        y = (~y.astype(str).str.lower().isin(["benign", "normal", "0"])).astype(int)
    df["label"] = y.astype(int).values
    ids = {"ts", "timestamp", "flow id", "flow_id", "src ip", "dst ip", "src_ip", "dst_ip", "source ip",
           "destination ip", "type", "attack", "label"} | {x.lower() for x in c["drop"]}
    df = df.drop(columns=[x for x in df.columns if x.lower() in ids and x != "label"])
    df = df.replace([np.inf, -np.inf], np.nan)
    n0 = len(df)
    df = df.drop_duplicates()
    feats = [x for x in df.columns if x != "label"]
    conflict = df.duplicated(feats, keep=False)          # same features, different label
    df = df.loc[~conflict].reset_index(drop=True)
    log.info(f"[ToN-IoT] {[f.name for f in files]}: {n0:,} rows -> {len(df):,} after removing "
             f"duplicates/conflicts | attack share {df['label'].mean():.2%}")
    cat = []
    for col in feats:
        if not pd.api.types.is_numeric_dtype(df[col]):
            num = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")
            if num.notna().mean() >= 0.95 * df[col].replace("-", np.nan).notna().mean():
                df[col] = num.astype("float32")
            else:
                cat.append(col)
        else:
            df[col] = df[col].astype("float32")
    y = df.pop("label").astype(int).values
    return df, y, cat


class OrdinalEncoder:
    """Category -> integer, fitted on training rows only; unseen categories -> -1."""

    def fit(self, X: pd.DataFrame, cols: list[str]):
        self.maps = {c: {v: i for i, v in enumerate(X[c].astype(str).unique())} for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, m in self.maps.items():
            X[c] = X[c].astype(str).map(m).fillna(-1).astype("float32")
        return X


# ============================================================================ CICIoMT2024
def _label(fname: str) -> str:
    stem = Path(fname).name
    stem = re.split(r"_(train|test)", stem, maxsplit=1, flags=re.I)[0]
    return re.sub(r"\d+$", "", stem)


def _category(label: str) -> str:
    l = label.lower()
    if "benign" in l:
        return "Benign"
    if "mqtt" in l:
        return "MQTT"
    if "spoof" in l:
        return "Spoofing"
    if "recon" in l:
        return "Recon"
    if "ddos" in l:
        return "DDoS"
    if "dos" in l:
        return "DoS"
    return "Other"


def is_device_insider(label: str) -> bool:
    """Attacks a compromised *registered* device performs from inside the network."""
    l = label.lower()
    return "spoof" in l or "malformed" in l or "publish" in l


def prepare_ciciomt(cfg: dict) -> tuple[Path, Path]:
    c = cfg["iot"]["ciciomt2024"]
    out_tr = cfg["paths"]["features"] / "ciciomt_train.parquet"
    out_te = cfg["paths"]["features"] / "ciciomt_test.parquet"
    if done(out_te):
        log.info("[CICIoMT2024] prepared - skip")
        return out_tr, out_te
    root = Path(c["dir"])
    root = root if root.is_absolute() else ROOT / root
    files = [p for p in root.rglob("*.csv") if re.search(r"(train|test)", str(p), re.I)
             and "profiling" not in str(p).lower()]
    if not files:
        raise FileNotFoundError(f"no CICIoMT2024 CSVs under {root} - see README")
    parts = {"train": [], "test": []}
    for fp in sorted(files):
        split = "test" if re.search(r"test", fp.name, re.I) or "/test/" in str(fp).lower() else "train"
        df = pd.read_csv(fp, low_memory=False)
        df.columns = [x.strip() for x in df.columns]
        df = df.drop(columns=[x for x in c["drop"] if x in df.columns])
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
        lab = _label(fp.name)
        df["label_name"], df["category"] = lab, _category(lab)
        parts[split].append(df)
        log.info(f"[CICIoMT2024] {split:5s} {fp.name}: {len(df):,} rows -> {lab}")
    tr = pd.concat(parts["train"], ignore_index=True).drop_duplicates()
    te = pd.concat(parts["test"], ignore_index=True).drop_duplicates()
    feats = [x for x in tr.columns if x not in ("label_name", "category")]
    # remove test rows that also occur in training (cross-split duplicates inflate scores)
    h_tr = pd.util.hash_pandas_object(tr[feats], index=False)
    h_te = pd.util.hash_pandas_object(te[feats], index=False)
    n_before = len(te)
    te = te.loc[~h_te.isin(set(h_tr.values))].reset_index(drop=True)
    # cap per class in training (speed); sampling is fixed (seed 0) so it is identical for every run
    cap = int(c["max_per_class"])
    tr = tr.sample(frac=1.0, random_state=0).reset_index(drop=True)
    tr = tr.loc[tr.groupby("label_name").cumcount() < cap].reset_index(drop=True)
    save_parquet(tr, out_tr)
    save_parquet(te, out_te)
    mark_done(out_te)
    log.info(f"[CICIoMT2024] train {len(tr):,} | test {len(te):,} "
             f"({n_before - len(te):,} test rows duplicated in train removed)")
    return out_tr, out_te
