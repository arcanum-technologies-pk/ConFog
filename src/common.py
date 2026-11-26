"""Shared utilities: config, logging, atomic saving, resume markers, seeding, timing."""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("itd")


# ----------------------------------------------------------------------------- config
def _merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def load_config(path: str | Path = ROOT / "config.yaml", override: str | Path | None = None) -> dict:
    with open(ROOT / path) as f:
        cfg = yaml.safe_load(f)
    if override:
        with open(ROOT / override) as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


# ----------------------------------------------------------------------------- logging
def setup_logging(log_dir: Path, name: str = "run") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("itd")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


# ----------------------------------------------------------------------------- atomic IO
def _atomic(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)  # atomic on POSIX: a crash never leaves a half-written file


def save_json(obj, path) -> None:
    _atomic(path, lambda p: Path(p).write_text(json.dumps(obj, indent=2, default=_np_default)))


def load_json(path):
    return json.loads(Path(path).read_text())


def save_pickle(obj, path) -> None:
    def w(p):
        with open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic(path, w)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df, path) -> None:
    _atomic(path, lambda p: df.to_parquet(p, index=False))


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ----------------------------------------------------------------------------- resume markers
def done(path) -> bool:
    return Path(str(path) + ".done").exists()


def mark_done(path) -> None:
    Path(str(path) + ".done").parent.mkdir(parents=True, exist_ok=True)
    Path(str(path) + ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ----------------------------------------------------------------------------- misc
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "cuda"):
    import torch
    return torch.device("cuda" if pref == "cuda" and torch.cuda.is_available() else "cpu")


@contextmanager
def timer(label: str, store: dict | None = None, key: str | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[key or label] = dt
    log.info(f"{label} finished in {fmt_s(dt)}")


def fmt_s(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"
