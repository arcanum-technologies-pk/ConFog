"""Train / validation / test splits for CERT user-days (row positions into the release frame).

Warm-up rows (first days of a release) are never in any split; they only serve as history.
Validation data always comes from the training side, the test side is touched once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _users_split(users: np.ndarray, insider: np.ndarray, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (insider / normal) user split -> (users_a, users_b) with |b| ~= frac."""
    b = []
    for flag in (True, False):
        u = users[insider == flag]
        u = u[rng.permutation(len(u))]
        k = int(round(frac * len(u)))
        if flag and frac > 0 and len(u) >= 2:
            k = min(max(k, 1), len(u) - 1)   # keep >=1 insider on both sides
        b.append(u[:k])
    b = np.concatenate(b)
    return np.setdiff1d(users, b), b


def _user_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df.groupby("user")["scenario"].max()
    return t.index.values, (t.values > 0)


def _rows(df: pd.DataFrame, mask) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask) & ~df["warmup"].values)


def _time_cut(df: pd.DataFrame, frac_before: float, rows=None) -> np.datetime64:
    """Day such that ~frac_before of the (eligible) user-days lie on or before it."""
    days = np.sort(df["day"].values if rows is None else df["day"].values[rows])
    return days[int(frac_before * (len(days) - 1))]


def chrono(df: pd.DataFrame, test_frac: float, seed: int, val_frac_time: float | None = None,
           val_frac_users: float | None = None) -> dict:
    elig = _rows(df, True)
    cut = _time_cut(df, 1 - test_frac, elig)
    day = df["day"].values
    train = elig[day[elig] <= cut]
    test = elig[day[elig] > cut]
    if val_frac_time:  # protocol of [11]: last part of the training period
        vcut = _time_cut(df, 1 - val_frac_time, train)
        val, train = train[day[train] > vcut], train[day[train] <= vcut]
    else:              # protocol of [13]: hold-out users from the training period
        users, ins = _user_table(df)
        _, vu = _users_split(users, ins, val_frac_users, np.random.default_rng(seed))
        inval = np.isin(df["user"].values[train], vu)
        val, train = train[inval], train[~inval]
    return {"train": train, "val": val, "test": test}


def user_disjoint(df: pd.DataFrame, test_frac: float, val_frac_users: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    users, ins = _user_table(df)
    tr_u, te_u = _users_split(users, ins, test_frac, rng)
    sel = np.isin(users, tr_u)
    tr_u, va_u = _users_split(users[sel], ins[sel], val_frac_users, rng)
    u = df["user"].values
    return {"train": _rows(df, np.isin(u, tr_u)), "val": _rows(df, np.isin(u, va_u)),
            "test": _rows(df, np.isin(u, te_u))}


def leave_one_scenario_out(df: pd.DataFrame, scenario: int, test_normal_frac: float,
                           val_frac_users: float, seed: int) -> dict:
    """Zero-day protocol: insiders of `scenario` are never seen in training or validation."""
    rng = np.random.default_rng(seed)
    t = df.groupby("user")["scenario"].max()
    held = t.index.values[t.values == scenario]
    normal = t.index.values[t.values == 0]
    others = t.index.values[(t.values > 0) & (t.values != scenario)]
    normal = normal[rng.permutation(len(normal))]
    te_norm = normal[: int(round(test_normal_frac * len(normal)))]
    tr_users = np.concatenate([np.setdiff1d(normal, te_norm), others])
    ins = np.isin(tr_users, others)
    tr_u, va_u = _users_split(tr_users, ins, val_frac_users, rng)
    u = df["user"].values
    return {"train": _rows(df, np.isin(u, tr_u)), "val": _rows(df, np.isin(u, va_u)),
            "test": _rows(df, np.isin(u, np.concatenate([held, te_norm])))}


def make_split(df: pd.DataFrame, protocol: str, cfg: dict, seed: int, scenario: int | None = None,
               source_only: bool = False) -> dict:
    """source_only: release used only to train cross-dataset models (r6.2, as in [13]) -> no test part."""
    s = cfg["split"]
    if protocol == "chrono80":
        sp = chrono(df, s["chrono80"]["test_frac"], seed, val_frac_time=s["chrono80"]["val_frac_time"])
    elif protocol == "chrono70":
        sp = chrono(df, s["chrono70"]["test_frac"], seed, val_frac_users=s["chrono70"]["val_frac_users"])
    elif protocol == "userdisj70":
        sp = user_disjoint(df, 0.0 if source_only else s["userdisj70"]["test_frac"],
                           s["userdisj70"]["val_frac_users"], seed)
    elif protocol == "loso":
        sp = leave_one_scenario_out(df, scenario, s["loso"]["test_normal_frac"], s["loso"]["val_frac_users"], seed)
    else:
        raise ValueError(protocol)
    check_split(df, sp, protocol)
    return sp


def check_split(df: pd.DataFrame, sp: dict, protocol: str) -> None:
    """Hard guards against leakage between partitions."""
    a, b, c = (set(sp[k].tolist()) for k in ("train", "val", "test"))
    assert not (a & b or a & c or b & c), "row overlap between partitions"
    if protocol in ("userdisj70", "loso"):
        u = df["user"].values
        ua, ub, uc = set(u[sp["train"]]), set(u[sp["val"]]), set(u[sp["test"]])
        assert not (ua & uc or ub & uc), "user appears in train/val and test"
    if protocol.startswith("chrono"):
        d = df["day"].values
        assert d[sp["train"]].max() < d[sp["test"]].min(), "training days overlap test period"
    if len(sp["train"]) == 0 or len(sp["val"]) == 0:
        raise ValueError(f"{protocol}: empty train or validation split")
    for k in ("train", "val", "test"):
        if len(sp[k]) and df["label"].values[sp[k]].sum() == 0:
            raise ValueError(f"{protocol}: no positive user-days in {k} split")
