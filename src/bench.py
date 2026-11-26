"""System benchmark (RQ1 / RQ4): fog gateways in Docker (1 CPU, 512 MB, tc-netem link) + cloud
service (GRU + graph model on the GPU) reached over the network.

For every scenario (link RTT / bandwidth, CPU contention on the fog) and every forwarding mode
(adaptive / static / cloud_only) it measures detection quality, end-to-end latency, uplink bytes,
fog CPU/memory and energy:
  * CPU energy  : Intel/AMD RAPL counters (measured), idle power subtracted, attributed by CPU time
  * GPU energy  : NVIDIA NVML total-energy counter (measured)
If a counter is not readable the value is marked "estimated" (CPU time x watts_per_core).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from .common import ROOT, get_device, load_json, load_pickle, log, save_json
from .graph import graph_inputs
from .metrics import binary_report
from .models import SeqData, _SeqGraph

IMAGE = "itd-fog:latest"


# ============================================================================ energy counters
class Rapl:
    def __init__(self):
        base = Path("/sys/class/powercap")
        self.zones = [z for z in base.glob("intel-rapl:*") if ":" in z.name and z.name.count(":") == 1]
        try:
            self.ok = bool(self.zones) and all(os.access(z / "energy_uj", os.R_OK) and
                                               int((z / "energy_uj").read_text()) >= 0 for z in self.zones)
        except OSError:
            self.ok = False
        self.max = [int((z / "max_energy_range_uj").read_text()) for z in self.zones] if self.ok else []

    def read(self):
        return [int((z / "energy_uj").read_text()) for z in self.zones] if self.ok else []

    def joules(self, a, b):
        return sum(((y - x) % m if y < x else y - x) for x, y, m in zip(a, b, self.max)) / 1e6


class RaplMeter:
    """PATCH_ENERGY: polls RAPL every 0.2 s so that no counter wrap is missed."""
    def __init__(self, rapl, dt=0.2):
        self.r, self.dt, self.j, self.lock = rapl, dt, 0.0, threading.Lock()
        self.last = rapl.read()
        if rapl.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(self.dt)
            cur = self.r.read()
            with self.lock:
                self.j += self.r.joules(self.last, cur)
                self.last = cur

    def total(self):
        with self.lock:
            return self.j + self.r.joules(self.last, self.r.read())


class Nvml:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.f = pynvml.nvmlDeviceGetTotalEnergyConsumption
            self.f(self.h)
            self.ok = True
        except Exception:
            self.ok = False

    def read(self):
        return self.f(self.h) if self.ok else 0     # millijoules


# ============================================================================ cloud service
class Cloud:
    def __init__(self, cfg, mdir, df):
        dev = get_device(cfg["device"])
        ck = torch.load(mdir / "cloud_gru.pt", map_location="cpu", weights_only=False)
        b = load_pickle(mdir / "fog_bundle.pkl")
        g = cfg["models"]["gru"]
        self.net = _SeqGraph(ck["d"], ck.get("g_dim", 0), g["hidden"], g["dropout"]).to(dev).eval()
        self.net.load_state_dict(ck["state"])
        sc = b["scaler"]
        X = df[b["feats"]].values.astype("float32")
        Xn = np.clip((np.sign(X) * np.log1p(np.abs(X)) - sc["mu"]) / sc["sd"], -sc["clip"], sc["clip"]).astype("float32")
        G = np.hstack([Xn, graph_inputs(cfg, df.attrs["release"], Xn)]) if ck.get("g_dim") else None
        self.data = SeqData(Xn, df["user"].values, g["window"], dev, G)   # cloud keeps users' history
        self.w, self.dev, self.lock = ck["w"], dev, threading.Lock()
        for _ in range(3):                         # warm-up: CUDA context / kernels, excluded from timing
            self.score(np.arange(64), np.zeros(64), False)

    def score(self, rows, fog, cloud_only):
        t = time.perf_counter()
        with self.lock, torch.no_grad():
            r = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(self.dev)
            c = torch.sigmoid(self.net(self.data.batch(r), self.data.graph(r))).cpu().numpy()
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
        s = c if cloud_only else 1.0 + self.w * c + (1 - self.w) * np.asarray(fog)
        return s.tolist(), 1000 * (time.perf_counter() - t)


def _serve(cloud: Cloud, port: int) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            base64.b64decode(req["x"])                      # payload really transferred and decoded
            s, ms = cloud.score(req["rows"], req["fog"], req.get("cloud_only", False))
            body = json.dumps({"score": s, "compute_ms": ms}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ============================================================================ fog nodes
def _docker_ok(required: bool) -> bool:
    why = None
    if not shutil.which("docker"):
        why = "docker is not installed"
    elif subprocess.run(["docker", "info"], capture_output=True).returncode:
        why = "docker daemon not reachable (run with: sg docker -c 'python run.py --stage bench')"
    else:
        r = subprocess.run(["docker", "build", "--network=host", "-q", "-t", IMAGE, "-f", "docker/Dockerfile.fog", "."], cwd=ROOT,
                           capture_output=True, text=True)
        why = None if r.returncode == 0 else f"docker build failed: {r.stderr[-800:]}"
    if why and required:
        raise RuntimeError(f"bench: Docker required (config bench.require_docker) but {why}")
    if why:
        log.warning(f"bench: {why} -> local processes")
    return why is None


def _run_nodes(cfg, docker, bundle, data_dir, out_dir, sc, mode, port):
    b = cfg["bench"]
    args = ["--mode", mode, "--rate", str(b["rate_eps"]), "--batch", str(b["batch"]), "--stress", str(sc["stress"]),
            "--rtt-ms", str(sc["rtt_ms"]), "--mbit", str(sc["mbit"]), "--load-ref", str(b["adapt"]["load_ref"]),
            "--rtt-ref-ms", str(b["adapt"]["rtt_ref_ms"]), "--backlog-ref", str(b["adapt"]["backlog_ref"])]
    procs = []
    for k in range(b["n_fog_nodes"]):
        io = ["--events", f"/data/node_{k}.npz", "--out", f"/out/node_{k}.json"] if docker else \
             ["--events", str(data_dir / f"node_{k}.npz"), "--out", str(out_dir / f"node_{k}.json")]
        if docker:
            cmd = ["docker", "run", "--rm", f"--cpus={b['cpus']}", f"--memory={b['mem']}", "--cap-add=NET_ADMIN",
                   "--add-host=host.docker.internal:host-gateway", "-v", f"{bundle}:/bundle:ro",
                   "-v", f"{data_dir}:/data:ro", "-v", f"{out_dir}:/out", IMAGE, "--bundle", "/bundle",
                   "--cloud", f"http://host.docker.internal:{port}/", "--netem", "1"] + io + args
            procs.append(subprocess.Popen(cmd, cwd=ROOT))
        else:
            core = sorted(os.sched_getaffinity(0))[k % len(os.sched_getaffinity(0))]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            cmd = ["taskset", "-c", str(core), sys.executable, "-m", "src.fog_node", "--bundle", str(bundle),
                   "--cloud", f"http://127.0.0.1:{port}/"] + io + args
            procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
    for p in procs:
        if p.wait():
            raise RuntimeError("fog node failed")
    return [load_json(out_dir / f"node_{k}.json") for k in range(b["n_fog_nodes"])]


# ============================================================================ benchmark
def run_bench(cfg: dict) -> None:
    out = cfg["paths"]["results"] / "bench" / "system.json"
    if out.exists():
        log.info("bench: results exist - skip")
        return
    b = cfg["bench"]
    mdir = cfg["paths"]["models"] / "fog_main"
    meta = load_json(mdir / "test_rows.json")
    df = pd.read_parquet(cfg["paths"]["features"] / f"{meta['release']}.parquet")
    df.attrs["release"] = meta["release"]
    bundle = load_pickle(mdir / "fog_bundle.pkl")
    rows = np.array(meta["test_rows"])
    rows = np.sort(np.random.default_rng(0).choice(rows, min(b["n_events"], len(rows)), replace=False))
    X = df[bundle["feats"]].values[rows].astype("float32")
    y = df["label"].values[rows].astype(int)
    data_dir, node_dir = (out.parent / "data").resolve(), (out.parent / "nodes").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    node_dir.mkdir(parents=True, exist_ok=True)
    shards = np.array_split(np.arange(len(rows)), b["n_fog_nodes"])
    for k, s in enumerate(shards):
        np.savez(data_dir / f"node_{k}.npz", X=X[s], rows=rows[s])

    docker = _docker_ok(b.get("require_docker", True))
    mode_txt = "docker (--cpus 1, --memory 512m, tc-netem)" if docker else "local processes (1 core each, emulated link)"
    log.info(f"bench: {len(rows):,} events, {b['n_fog_nodes']} fog nodes, {mode_txt}")
    cloud = Cloud(cfg, mdir, df)
    srv = _serve(cloud, b["port"])
    rapl, nvml, me = Rapl(), Nvml(), psutil.Process()
    meter = RaplMeter(rapl)
    # idle power of CPU package (RAPL) and GPU (NVML) with the cloud model loaded; subtracted from every run
    e0, g0 = rapl.read(), nvml.read()
    time.sleep(b["idle_s"])
    p_idle = rapl.joules(e0, rapl.read()) / b["idle_s"] if rapl.ok else 0.0
    p_gpu_idle = (nvml.read() - g0) / 1000 / b["idle_s"] if nvml.ok else 0.0
    thr = bundle["thr"]
    role, off = df["role_group"].values[rows], df["ctx_offhours"].values[rows]
    ct = thr.get("context", {})
    ctx_t = np.array([ct.get(f"{r}|{o}", ct.get(f"{r}|None", thr["global"])) for r, o in zip(role, off)])

    runs = []
    plan = []
    for rep_i in range(b["repeats"]):              # every repeat runs all conditions in a new random order
        block = [(rep_i, sc, m) for sc in b["scenarios"] for m in b["modes"]]
        plan += [block[o] for o in np.random.default_rng(rep_i).permutation(len(block))]
    try:
        for rep_i, sc, mode in plan:
            if rapl.ok and b.get("idle_each_run"):          # idle power right before this run
                i0, ig0, it0 = meter.total(), nvml.read(), time.perf_counter()
                time.sleep(b["idle_s"])
                dti = time.perf_counter() - it0
                p_idle = (meter.total() - i0) / dti
                p_gpu_idle = (nvml.read() - ig0) / 1000 / dti if nvml.ok else 0.0
            c0, g0, me0, t0 = meter.total(), nvml.read(), me.cpu_times(), time.perf_counter()
            nodes = _run_nodes(cfg, docker, mdir.resolve(), data_dir, node_dir, sc, mode, b["port"])
            wall = time.perf_counter() - t0
            c1, g1, me1 = meter.total(), nvml.read(), me.cpu_times()
            score = np.concatenate([n["score"] for n in nodes])
            fwd = np.concatenate([n["fwd"] for n in nodes]).astype(bool)
            dec = np.concatenate([n["decision_ms"] for n in nodes]).astype(float)
            alert = np.concatenate([n["alert_ms"] for n in nodes]).astype(float)
            fog_cpu = sum(n["cpu_s"] for n in nodes)
            host_cpu = (me1.user - me0.user) + (me1.system - me0.system)
            stress_cpu = sum(n["stress_cpu_s"] for n in nodes)
            r = {"repeat": rep_i, "scenario": sc["name"], "mode": mode, "rtt_ms": sc["rtt_ms"], "mbit": sc["mbit"],
                 "stress": sc["stress"], "link": nodes[0]["link"], "forward_rate": float(fwd.mean()),
                 "decision_ms_p50": float(np.nanpercentile(dec, 50)), "decision_ms_p95": float(np.nanpercentile(dec, 95)),
                 "alert_ms_p95": float(np.nanpercentile(alert, 95)) if fwd.any() else None,
                 "uplink_kb": sum(n["bytes_up"] for n in nodes) / 1024,
                 "backlog_max": max(n["backlog_max"] for n in nodes),
                 "fog_cpu_pct": 100 * fog_cpu / (wall * len(nodes)),
                 "fog_mem_mb_peak": max(n["mem_mb_peak"] for n in nodes),
                 "tau_mean": float(np.mean([n["tau_mean"] for n in nodes])),
                 "fog_load_mean": float(np.mean([n["load_mean"] for n in nodes])),
                 "rtt_ms_measured": float(np.mean([n["rtt_ms_mean"] for n in nodes if n["rtt_ms_mean"] is not None] or [0]))}
            per1k = 1000 / len(rows) / 3.6                       # J -> mWh per 1k events
            if rapl.ok:
                r.update(p_idle_w=p_idle, cpu_j_raw=c1 - c0, wall_s=wall)
                e_cpu = max((c1 - c0) - p_idle * wall, 0.0)
                tot = max(fog_cpu + host_cpu + stress_cpu, 1e-9)
                r.update(energy_source="RAPL (measured)", fog_energy_mwh_per_1k=e_cpu * fog_cpu / tot * per1k,
                         cloud_cpu_energy_mwh_per_1k=e_cpu * host_cpu / tot * per1k)
            else:
                r.update(energy_source="estimated (CPU time x watts_per_core)",
                         fog_energy_mwh_per_1k=fog_cpu * b["watts_per_core"] * per1k,
                         cloud_cpu_energy_mwh_per_1k=host_cpu * b["watts_per_core"] * per1k)
            r["gpu_energy_mwh_per_1k"] = max((g1 - g0) / 1000 - p_gpu_idle * wall, 0.0) * per1k if nvml.ok else None
            r["total_energy_mwh_per_1k"] = (r["fog_energy_mwh_per_1k"] + r["cloud_cpu_energy_mwh_per_1k"]
                                            + (r["gpu_energy_mwh_per_1k"] or 0.0))
            if mode != "cloud_only" and y.sum():          # final decision: context-adaptive threshold
                pred = (fwd & (score >= ctx_t)).astype(int)
                m = binary_report(y, np.where(fwd, score, score - 1), pred)
                r.update({k2: m[k2] for k2 in ("recall", "precision", "f1", "fpr")}, malicious=int(y.sum()),
                         malicious_forwarded=int((fwd & (y == 1)).sum()))
            runs.append(r)
            log.info(f"bench [rep {rep_i} | {sc['name']:>15s} | {mode:>10s}] fwd {r['forward_rate']:.4f} | "
                     f"decision p95 {r['decision_ms_p95']:.1f} ms | alert p95 {r['alert_ms_p95'] or 0:.1f} ms | "
                     f"uplink {r['uplink_kb']:.1f} KB | backlog {r['backlog_max']} | tau {r['tau_mean']:.3f} | "
                     f"recall {r.get('recall', float('nan')):.3f} | energy {r['total_energy_mwh_per_1k']:.3f} mWh/1k")
    finally:
        srv.shutdown()
    save_json({"fog_mode": mode_txt, "n_events": int(len(rows)), "n_fog_nodes": b["n_fog_nodes"],
               "p_idle_w": p_idle, "p_gpu_idle_w": p_gpu_idle, "rapl": rapl.ok, "nvml": nvml.ok, "runs": runs}, out)
