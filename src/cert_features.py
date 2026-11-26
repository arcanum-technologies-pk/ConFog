"""Phase I - CERT r4.2 / r5.2 / r6.2  ->  user-day feature tables with behavioural, context and labels.

* Raw CSVs are read in fixed-size byte blocks; each block's partial aggregate is saved, so a crash
  resumes from the last finished block (no re-reading of finished blocks).
* All per-user history features are causal: day d only uses days < d (no look-ahead leakage).
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import done, fmt_s, load_json, log, mark_done, save_json, save_parquet

SOURCES = ["logon", "device", "file", "email", "http"]
DEFAULT_COLS = {  # used only if a file has no header row
    "logon": ["id", "date", "user", "pc", "activity"],
    "device": ["id", "date", "user", "pc", "activity"],
    "file": ["id", "date", "user", "pc", "filename", "content"],
    "email": ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "size", "attachments", "content"],
    "http": ["id", "date", "user", "pc", "url", "content"],
}
COUNTS = {
    "logon": ["logon_n", "logoff_n", "logon_off_n"],
    "device": ["usb_conn_n", "usb_off_n"],
    "file": ["file_n", "file_off_n", "file_exe_n", "file_doc_n", "file_zip_n"],
    "email": ["email_n", "email_off_n", "email_ext_n", "email_bcc_n", "email_att_n", "email_size_sum"],
    "http": ["http_n", "http_off_n", "http_job_n", "http_leak_n", "http_hack_n"],
}
PC_COUNTS = ["pc_n", "new_pc_n"]
HOUR_COLS = ["first_hour", "last_hour", "span_hours"]
ROLE_GROUPS = ["it_admin", "clinical_lead", "biomedical_engineer", "nurse", "staff"]


def count_cols() -> list[str]:
    return [c for s in SOURCES for c in COUNTS[s]] + PC_COUNTS


def modality_of(col: str) -> str | None:
    base = col[4:] if col.startswith("dev_") else col
    for s in SOURCES:
        if base in COUNTS[s]:
            return s
    if base in PC_COUNTS:
        return "logon"
    return None


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    behav = count_cols() + HOUR_COLS + ["off_ratio"]
    dev = [c for c in df.columns if c.startswith("dev_")]
    ctx = [c for c in df.columns if c.startswith("ctx_")]
    return {"behaviour": behav, "deviation": dev, "context": ctx}


# ============================================================================ block reader
def _blocks(path: Path, block_bytes: int, start: int):
    """Yield (start, end, bytes) blocks that end on a newline."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < size:
            buf = f.read(block_bytes)
            if not buf:
                break
            if pos + len(buf) < size:
                buf += f.readline()
            yield pos, pos + len(buf), buf
            pos += len(buf)


def _header(path: Path, src: str, release: str) -> tuple[list[str], int]:
    with open(path, "rb") as f:
        first = f.readline()
    tokens = first.decode("utf-8", "ignore").strip().split(",")
    if tokens and tokens[0].strip().lower() == "id":
        return [t.strip().lower() for t in tokens], len(first)
    cols = list(DEFAULT_COLS[src])
    if src == "http" and release != "r4.2":
        cols = ["id", "date", "user", "pc", "url", "activity", "content"]
    log.warning(f"{path.name}: no header row, assuming columns {cols}")
    return cols, 0


# ============================================================================ per-source aggregation
class Aggregator:
    def __init__(self, cfg: dict):
        c = cfg["cert"]
        self.w0, self.w1 = c["workday"]
        self.domain = re.escape(c["domain"])
        self.kw = {k: [w.lower() for w in v] for k, v in c["url_keywords"].items()}

    def _time(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        df = df.loc[dt.notna()].copy()
        dt = dt[dt.notna()]
        df["day"] = dt.dt.normalize()
        df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
        df["off"] = (dt.dt.dayofweek >= 5) | (df["hour"] < self.w0) | (df["hour"] >= self.w1)
        return df

    def __call__(self, src: str, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = self._time(df)
        f = pd.DataFrame({"user": df["user"].values, "day": df["day"].values,
                          "hmin": df["hour"].values, "hmax": df["hour"].values})
        off = df["off"].values
        out = {}
        if src == "logon":
            on = (df["activity"].str.lower() == "logon").values
            f["logon_n"], f["logoff_n"], f["logon_off_n"] = on, ~on, on & off
            out["logonpc"] = df[["user", "day", "pc"]].drop_duplicates()
        elif src == "device":
            con = (df["activity"].str.lower() == "connect").values
            f["usb_conn_n"], f["usb_off_n"] = con, con & off
        elif src == "file":
            fn = df["filename"].fillna("").str.lower()
            f["file_n"], f["file_off_n"] = 1, off
            f["file_exe_n"] = fn.str.endswith(".exe").values
            f["file_doc_n"] = fn.str.endswith((".doc", ".docx", ".pdf", ".txt")).values
            f["file_zip_n"] = fn.str.endswith((".zip", ".rar", ".7z")).values
        elif src == "email":
            if "activity" in df:  # r5+ logs both Send and View; r4.2 logs sent mail only
                keep = (df["activity"].str.lower() == "send").values
                df, f, off = df.loc[keep], f.loc[keep].reset_index(drop=True), off[keep]
            rcpt = (df["to"].fillna("") + ";" + df["cc"].fillna("") + ";" + df["bcc"].fillna(""))
            att = df["attachments"]
            n_att = pd.to_numeric(att, errors="coerce")
            s = att.fillna("").astype(str)
            alt = np.where(s.str.contains(r"\(", regex=True), s.str.count(r"\("),
                           np.where(s.str.len() > 0, s.str.count(";") + 1, 0))
            n_att = n_att.where(n_att.notna(), pd.Series(alt, index=att.index))
            f["email_n"], f["email_off_n"] = 1, off
            f["email_ext_n"] = rcpt.str.contains(rf"@(?!{self.domain})[\w.-]+", regex=True).values
            f["email_bcc_n"] = (df["bcc"].fillna("").str.len() > 0).values
            f["email_att_n"] = n_att.values
            f["email_size_sum"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).values
        elif src == "http":
            parts = df["url"].fillna("").str.lower().str.split("/", n=4, expand=True)
            empty = pd.Series("", index=parts.index)
            dom = parts[2].fillna("") if 2 in parts else empty
            seg = parts[3].fillna("") if 3 in parts else empty
            key = (dom + "/" + seg).astype("category")  # domain + first path segment
            cats = key.cat.categories
            f["http_n"], f["http_off_n"] = 1, off
            for k in ("job", "leak", "hack"):
                flag = np.array([any(w in c for w in self.kw[k]) for c in cats] + [False])
                f[f"http_{k}_n"] = flag[key.cat.codes.values]
        agg = {c: "sum" for c in f.columns if c.endswith("_n") or c.endswith("_sum")}
        agg.update(hmin="min", hmax="max")
        out["agg"] = f.groupby(["user", "day"], sort=False).agg(agg).reset_index()
        return out


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    agg = {c: ("min" if c == "hmin" else "max" if c == "hmax" else "sum")
           for c in df.columns if c not in ("user", "day")}
    return df.groupby(["user", "day"], sort=False).agg(agg).reset_index()


def aggregate_source(cfg: dict, release: str, src: str) -> None:
    raw = cfg["paths"]["raw"] / "cert" / release / f"{src}.csv"
    work = cfg["paths"]["work"] / "cert" / release / src
    final = work.parent / f"{src}.parquet"
    if done(final):
        log.info(f"[{release}][{src}] already aggregated - skip")
        return
    if not raw.exists():
        raise FileNotFoundError(f"{raw} not found - run scripts/download_cert.sh first")
    work.mkdir(parents=True, exist_ok=True)
    state_f = work / "state.json"
    names, data_start = _header(raw, src, release)
    start = load_json(state_f)["next_offset"] if state_f.exists() else data_start
    usecols = [c for c in names if c not in ("id", "content")]
    agg, size, t0 = Aggregator(cfg), raw.stat().st_size, time.time()
    block = int(cfg["cert"]["block_mb"]) * 2 ** 20
    if start > data_start:
        log.info(f"[{release}][{src}] resuming at byte {start:,} ({start / size:.1%})")
    for b0, b1, buf in _blocks(raw, block, start):
        df = pd.read_csv(io.BytesIO(buf), header=None, names=names, usecols=usecols, dtype=str,
                         engine="c", on_bad_lines="skip")
        skipped = buf.count(b"\n") - len(df)
        res = agg(src, df)
        for k, v in res.items():
            save_parquet(v, work / f"{k}_{b0:015d}.parquet")
        save_json({"next_offset": b1}, state_f)
        el = time.time() - t0
        frac = (b1 - start) / max(size - start, 1)
        log.info(f"[{release}][{src}] {b1 / 2**30:6.2f}/{size / 2**30:.2f} GB ({b1 / size:5.1%}) | "
                 f"rows {len(df):,} skipped {max(skipped, 0)} | elapsed {fmt_s(el)} "
                 f"ETA {fmt_s(el / max(frac, 1e-9) - el)}")
    save_parquet(_combine([pd.read_parquet(p) for p in sorted(work.glob("agg_*.parquet"))]), final)
    if src == "logon":
        pcs = pd.concat([pd.read_parquet(p) for p in sorted(work.glob("logonpc_*.parquet"))])
        save_parquet(pcs.drop_duplicates(), work.parent / "logonpc.parquet")
    mark_done(final)
    log.info(f"[{release}][{src}] aggregation complete")


# ============================================================================ labels, roles
def load_labels(cfg: dict, release: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (malicious user-days, insider users with scenario) from the official answer key."""
    ans = cfg["paths"]["raw"] / "cert" / "answers"
    ins = pd.read_csv(ans / "insiders.csv")
    ins = ins[np.isclose(pd.to_numeric(ins["dataset"], errors="coerce"), float(release[1:]))]
    if ins.empty:
        raise FileNotFoundError(f"no insiders for {release} in {ans / 'insiders.csv'}")
    index = {q.name: q for q in ans.rglob("*.csv") if q.name != "insiders.csv"}
    rows, missing = [], []
    for _, r in ins.iterrows():
        fp = index.get(str(r["details"]).strip())
        if fp is None:
            missing.append(r["details"])
            continue
        user = str(r["user"]).strip()
        for line in fp.read_text(errors="ignore").splitlines():
            f = [x.strip().strip('"') for x in line.split(",", 4)]   # r6.x answer files quote fields
            if len(f) >= 4 and f[3] == user:
                rows.append((f[3], f[2], int(r["scenario"])))
    log.info(f"[{release}] answer key: {len(ins)} insiders, {len(ins) - len(missing)} event files found, "
             f"{len(rows):,} malicious events")
    if missing:
        log.warning(f"[{release}] {len(missing)} insider event files not found, e.g. {missing[:3]}")
    if not rows:
        raise FileNotFoundError(f"no malicious events parsed for {release} under {ans}")
    ev = pd.DataFrame(rows, columns=["user", "date", "scenario"])
    ev["day"] = pd.to_datetime(ev["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce").dt.normalize()
    ev = ev.dropna(subset=["day"])
    days = ev[["user", "day"]].drop_duplicates().assign(label=1)
    users = ev.groupby("user")["scenario"].first().reset_index()
    return days, users


def load_roles(cfg: dict, release: str) -> pd.DataFrame:
    ldap = cfg["paths"]["raw"] / "cert" / release / "LDAP"
    files = sorted(ldap.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"LDAP folder missing: {ldap}")
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates("user_id", keep="first")[["user_id", "role"]].rename(columns={"user_id": "user"})
    rmap = cfg["cert"]["role_map"]

    def group(role: str) -> str:
        r = str(role).lower()
        for g in ROLE_GROUPS[:-1]:
            if any(k in r for k in rmap[g]):
                return g
        return "staff"
    df["role_group"] = df["role"].map(group)
    return df[["user", "role_group"]]


# ============================================================================ feature building
def _causal_dev(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """(x - mean of the user's previous days) / (std of previous days + 1); 0 on first day."""
    g = df.groupby("user", sort=False)
    x = df[cols].astype("float64")
    n = g.cumcount().values[:, None].astype("float64")
    cs = x.groupby(df["user"], sort=False).cumsum().values - x.values
    cs2 = (x ** 2).groupby(df["user"], sort=False).cumsum().values - x.values ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = cs / n
        std = np.sqrt(np.maximum(cs2 / n - mean ** 2, 0))
        dev = (x.values - mean) / (std + 1.0)
    dev[n[:, 0] == 0] = 0.0
    return pd.DataFrame(dev.astype("float32"), columns=[f"dev_{c}" for c in cols], index=df.index), mean


def build_features(cfg: dict, release: str) -> Path:
    out = cfg["paths"]["features"] / f"{release}.parquet"
    if done(out):
        log.info(f"[{release}] features exist - skip")
        return out
    for s in SOURCES:
        aggregate_source(cfg, release, s)
    work = cfg["paths"]["work"] / "cert" / release
    log.info(f"[{release}] merging sources and building causal features")
    df = None
    for s in SOURCES:
        a = pd.read_parquet(work / f"{s}.parquet")
        a = a.rename(columns={"hmin": f"hmin_{s}", "hmax": f"hmax_{s}"})
        df = a if df is None else df.merge(a, on=["user", "day"], how="outer")
    for c in count_cols():
        if c in df:
            df[c] = df[c].fillna(0).astype("float32")
    df["first_hour"] = df[[f"hmin_{s}" for s in SOURCES]].min(axis=1).astype("float32")
    df["last_hour"] = df[[f"hmax_{s}" for s in SOURCES]].max(axis=1).astype("float32")
    df = df.drop(columns=[c for c in df.columns if c.startswith(("hmin_", "hmax_"))])
    df["span_hours"] = df["last_hour"] - df["first_hour"]

    # PCs used that day; "new" = first time this user is ever seen on that PC (causal by definition)
    pcs = pd.read_parquet(work / "logonpc.parquet")
    first = pcs.groupby(["user", "pc"])["day"].transform("min")
    pcs["new"] = (pcs["day"] == first).astype("float32")
    pc = pcs.groupby(["user", "day"]).agg(pc_n=("pc", "size"), new_pc_n=("new", "sum")).reset_index()
    df = df.merge(pc, on=["user", "day"], how="left").fillna({"pc_n": 0, "new_pc_n": 0})
    df = df.sort_values(["user", "day"], kind="stable").reset_index(drop=True)

    offc = ["logon_off_n", "usb_off_n", "file_off_n", "email_off_n", "http_off_n"]
    tot = ["logon_n", "usb_conn_n", "file_n", "email_n", "http_n"]
    df["off_ratio"] = (df[offc].sum(axis=1) / (df[tot].sum(axis=1) + 1)).astype("float32")

    dev_cols = count_cols() + ["first_hour", "last_hour"]
    dev, mean_past = _causal_dev(df, dev_cols)
    df = pd.concat([df, dev], axis=1)

    # ---- context (behavioural / environmental / trust) ----
    roles = load_roles(cfg, release)
    df = df.merge(roles, on="user", how="left").fillna({"role_group": "staff"})
    for g in ROLE_GROUPS:
        df[f"ctx_role_{g}"] = (df["role_group"] == g).astype("float32")
    usual_start = mean_past[:, dev_cols.index("first_hour")]
    w0, w1 = cfg["cert"]["workday"]
    df["ctx_night_shift"] = np.nan_to_num((usual_start < w0) | (usual_start >= w1)).astype("float32")
    df["ctx_shift_offset"] = np.nan_to_num(df["first_hour"].values - usual_start).astype("float32")
    df["ctx_new_device"] = (df["new_pc_n"] > 0).astype("float32")
    df["ctx_offhours"] = (df[offc].sum(axis=1) > 0).astype("float32")
    df["ctx_weekend"] = (df["day"].dt.dayofweek >= 5).astype("float32")
    df["ctx_tenure"] = np.log1p(df.groupby("user", sort=False).cumcount()).astype("float32")

    # ---- labels from the official answer key ----
    mal, insiders = load_labels(cfg, release)
    df = df.merge(mal, on=["user", "day"], how="left").fillna({"label": 0})
    df = df.merge(insiders, on="user", how="left").fillna({"scenario": 0})
    df["label"] = df["label"].astype("int8")
    df["scenario"] = df["scenario"].astype("int8")
    df["warmup"] = df["day"] < df["day"].min() + pd.Timedelta(days=cfg["cert"]["warmup_days"])
    df = df.sort_values(["user", "day"], kind="stable").reset_index(drop=True)
    save_parquet(df, out)
    mark_done(out)
    pos = int(df.loc[~df["warmup"], "label"].sum())
    log.info(f"[{release}] features: {len(df):,} user-days, {df['user'].nunique()} users, "
             f"{pos} malicious user-days, {int((insiders['scenario'] > 0).sum())} insiders")
    return out
