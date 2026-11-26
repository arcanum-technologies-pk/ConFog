"""Phase III graph learning: temporal user-PC co-usage graph.

Two users become neighbours once they have both logged on to the same PC; the link exists only
from the day after it was first observed (causal). For user-day i, its neighbours are the most
recent *earlier* user-days of the linked users. The cloud model then learns on multi-hop neighbour
aggregations (SIGN-style graph learning: X, A X, A^2 X), fused with the GRU sequence encoder.
The neighbour structure is cached per release as a sparse row-normalised matrix A.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import done, log, mark_done


def build_graph(cfg: dict, release: str) -> Path:
    out = cfg["paths"]["features"] / f"graph_{release}.npz"
    if done(out):
        log.info(f"[{release}] graph exists - skip")
        return out
    g = cfg["models"]["graph"]
    df = pd.read_parquet(cfg["paths"]["features"] / f"{release}.parquet", columns=["user", "day"])
    pcs = pd.read_parquet(cfg["paths"]["work"] / "cert" / release / "logonpc.parquet")
    fu = pcs.groupby(["user", "pc"], as_index=False)["day"].min()              # first use of a PC by a user
    busy = fu.groupby("pc")["user"].nunique()
    fu = fu[fu["pc"].isin(busy.index[busy <= g["max_users_per_pc"]])]           # drop hub machines
    e = fu.merge(fu, on="pc", suffixes=("_u", "_v"))
    e = e[e["user_u"] != e["user_v"]]
    e["link"] = np.maximum(e["day_u"].values, e["day_v"].values)                # both have used it
    e = e.groupby(["user_u", "user_v"], as_index=False)["link"].min()
    e = e.sort_values(["user_u", "link", "user_v"]).groupby("user_u").head(g["max_neighbors"])

    rows = df.reset_index().rename(columns={"index": "i"})
    m = rows.merge(e, left_on="user", right_on="user_u")
    m = m.loc[m["link"] < m["day"], ["i", "user_v", "day"]]                     # link known before day d
    right = rows.rename(columns={"i": "j", "user": "user_v", "day": "day_j"}).sort_values("day_j")
    m = pd.merge_asof(m.sort_values("day"), right, left_on="day", right_on="day_j", by="user_v",
                      allow_exact_matches=False, direction="backward")         # neighbour's latest earlier day
    m = m.dropna(subset=["j"])
    i, j = m["i"].values.astype(np.int64), m["j"].values.astype(np.int64)
    n = len(df)
    deg = np.bincount(i, minlength=n).astype("float32")
    A = sp.csr_matrix((1.0 / deg[i], (i, j)), shape=(n, n), dtype=np.float32)
    sp.save_npz(out, A)
    mark_done(out)
    log.info(f"[{release}] graph: {len(e):,} user links, {A.nnz:,} temporal edges, "
             f"{(deg > 0).mean():.1%} of user-days have neighbours")
    return out


def graph_inputs(cfg: dict, release: str, Xn: np.ndarray) -> np.ndarray:
    """[A X, A^2 X, X - A X (deviation from graph neighbours), has-neighbours, log(1+degree)] per user-day
    (X already scaled with training statistics)."""
    A = sp.load_npz(cfg["paths"]["features"] / f"graph_{release}.npz")
    a1 = A @ Xn
    a2 = A @ a1
    deg = np.diff(A.indptr).astype("float32")[:, None]
    has = (deg > 0).astype("float32")
    return np.hstack([a1, a2, (Xn - a1) * has, has, np.log1p(deg)]).astype("float32")
