"""One fog gateway (runs inside a Docker container limited to 1 CPU / 512 MB).

Events arrive as a stream at a fixed rate. Each batch is scored locally (the fog decision); events
above the forwarding threshold are queued for the cloud and sent by a separate uplink thread, so a
slow network never blocks local detection (store-and-forward gateway).

Environmental context used by the adaptive policy (Objective 2 / RQ2):
  * fog load       = scoring time / time budget of the batch (EWMA)
  * link RTT       = measured round-trip time of the last cloud request
  * uplink backlog = batches waiting in the uplink queue
  tau = tau_fwd + (tau_max - tau_fwd) * clip(over-load + over-latency + over-backlog, 0, 1)
  (tau_fwd keeps 95 % validation recall, tau_max keeps 90 %: both chosen on validation data)

Link conditions are real: `tc netem` on the container interface (delay + bandwidth limit).
Without Docker the same conditions are emulated in the uplink thread (stated in the output).
"""
import argparse
import base64
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import threading
import time
import urllib.request

import numpy as np
import psutil

from src.fog_runtime import FogRuntime


def _busy():                                 # CPU contention from co-located workloads
    while True:
        pass


def _netem(rtt_ms: float, mbit: float) -> str:
    if not shutil.which("tc") or os.geteuid() != 0:
        return "client-emulated"
    subprocess.run(["tc", "qdisc", "del", "dev", "eth0", "root"], capture_output=True)
    r = subprocess.run(["tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", f"{rtt_ms}ms",
                        "rate", f"{mbit}mbit"], capture_output=True, text=True)
    return "tc-netem" if r.returncode == 0 else "client-emulated"


class Uplink(threading.Thread):
    """Sends forwarded batches to the cloud in order; records completion time and cloud score."""

    def __init__(self, url, link, rtt_ms, mbit, done, score):
        super().__init__(daemon=True)
        self.q, self.url, self.link, self.rtt_ms, self.mbit = queue.Queue(), url, link, rtt_ms, mbit
        self.done, self.score, self.rtt, self.rtts, self.bytes = done, score, rtt_ms, [], 0

    def post(self, body: bytes) -> dict:
        if self.link == "client-emulated":
            time.sleep(self.rtt_ms / 1000 + len(body) * 8 / (self.mbit * 1e6))
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=300).read())

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            idx, body = item
            t = time.perf_counter()
            resp = self.post(body)
            self.rtt = 1000 * (time.perf_counter() - t) - resp["compute_ms"]
            self.rtts.append(self.rtt)
            self.bytes += len(body)
            self.score[idx] = resp["score"]
            self.done[idx] = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    for a, t, d in [("--bundle", str, None), ("--events", str, None), ("--out", str, None), ("--cloud", str, None),
                    ("--mode", str, "adaptive"), ("--rate", float, 2000), ("--batch", int, 50), ("--stress", int, 0),
                    ("--rtt-ms", float, 0), ("--mbit", float, 1000), ("--load-ref", float, 0.7),
                    ("--rtt-ref-ms", float, 50), ("--backlog-ref", float, 4), ("--netem", int, 0)]:
        ap.add_argument(a, type=t, default=d)
    a = ap.parse_args()

    if not (a.rtt_ms or a.mbit < 1000):
        link = "none"
    else:   # real shaping only inside the container (--netem 1); never touch the host's interfaces
        link = _netem(a.rtt_ms, a.mbit) if a.netem else "client-emulated"
    proc = psutil.Process()
    fog = FogRuntime(a.bundle)
    ev = np.load(a.events)
    X, rows = ev["X"], ev["rows"]
    n = len(X)
    fog_score, final = np.zeros(n), np.zeros(n)
    fwd = np.zeros(n, bool)
    arrive, decided, confirmed = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    up = Uplink(a.cloud, link, a.rtt_ms, a.mbit, confirmed, final)

    # warm-up (model caches + HTTP connection), excluded from all measurements
    fog.score(X[: a.batch])
    up.post(json.dumps({"rows": rows[:1].tolist(), "fog": [0.0], "cloud_only": False,
                        "x": base64.b64encode(X[:1].tobytes()).decode()}).encode())
    stress = [mp.Process(target=_busy, daemon=True) for _ in range(a.stress)]
    for p in stress:
        p.start()
    up.start()

    taus, loads, backlog_max, load = [], [], 0, 0.0
    budget = a.batch / a.rate
    rss = proc.memory_info().rss
    cpu0 = proc.cpu_times()
    t0 = time.perf_counter()
    for b0 in range(0, n, a.batch):
        sl = slice(b0, min(b0 + a.batch, n))
        arrival = t0 + (b0 // a.batch) * budget
        wait = arrival - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        start = time.perf_counter()
        backlog = up.q.qsize()
        backlog_max = max(backlog_max, backlog)
        if a.mode == "cloud_only":            # baseline: no fog analytics, every record goes to the cloud
            s, tau = np.zeros(sl.stop - sl.start), -np.inf
        else:
            s = fog.score(X[sl])
            tau = fog.tau_fwd
            if a.mode == "adaptive":
                over = (max(0.0, load - a.load_ref) / (1 - a.load_ref)
                        + max(0.0, up.rtt - a.rtt_ref_ms) / a.rtt_ref_ms + backlog / a.backlog_ref)
                tau += (fog.tau_max - fog.tau_fwd) * min(1.0, over)
        f = s >= tau
        now = time.perf_counter()
        idx = np.arange(sl.start, sl.stop)
        fog_score[idx], final[idx], fwd[idx], arrive[idx], decided[idx] = s, s, f, arrival, now
        if f.any():
            body = json.dumps({"rows": rows[idx[f]].tolist(), "fog": s[f].tolist(), "cloud_only": a.mode == "cloud_only",
                               "x": base64.b64encode(X[idx[f]].astype("float32").tobytes()).decode()}).encode()
            up.q.put((idx[f], body))
        load = 0.7 * load + 0.3 * ((now - start) / budget)
        taus.append(float(tau) if np.isfinite(tau) else 0.0)
        loads.append(load)
        rss = max(rss, proc.memory_info().rss)
    up.q.put(None)
    up.join()
    wall = time.perf_counter() - t0
    cpu1 = proc.cpu_times()
    stress_cpu = 0.0
    for p in stress:
        try:
            ct = psutil.Process(p.pid).cpu_times()
            stress_cpu += ct.user + ct.system
        except psutil.Error:
            pass
        p.terminate()
    # decision latency: cloud_only decides in the cloud, the fog decides locally
    dec = confirmed if a.mode == "cloud_only" else decided
    res = {"n": n, "mode": a.mode, "link": link, "wall_s": wall,
           "cpu_s": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system), "stress_cpu_s": stress_cpu,
           "mem_mb_peak": rss / 2 ** 20, "bytes_up": up.bytes, "tau_mean": float(np.mean(taus)),
           "load_mean": float(np.mean(loads)), "backlog_max": int(backlog_max),
           "rtt_ms_mean": float(np.mean(up.rtts)) if up.rtts else None,
           "score": final.tolist(), "fwd": fwd.astype(int).tolist(),
           "decision_ms": (1000 * (dec - arrive)).tolist(),
           "alert_ms": np.where(fwd, 1000 * (confirmed - arrive), np.nan).tolist()}
    with open(a.out, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main()
