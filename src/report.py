"""Builds all comparison tables (Markdown + CSV) and figures from the saved raw results.
Values are mean +/- std over seeds. Base-paper rows are the numbers those papers report;
"reproduced" rows are the base methods re-run by this code on exactly the same splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import log
from .metrics import paired_bootstrap


def _load(dirpath: Path) -> pd.DataFrame:
    rows = []
    for f in dirpath.rglob("*.json"):
        d = json.loads(f.read_text())
        rows += d if isinstance(d, list) else [d]
    return pd.DataFrame(rows)


def _ms(x: pd.Series) -> str:
    x = x.dropna()
    return "-" if x.empty else (f"{x.mean():.4f}" if len(x) == 1 else f"{x.mean():.4f} ± {x.std(ddof=1):.4f}")


def _table(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    return df.groupby(by, sort=False)[metrics].agg(_ms).reset_index()


def _md(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    return "\n".join(lines)


NAMES = {"ours": "Proposed (fog-cloud, context-aware)", "base11": "[11] Alkharabsheh 2026 (reproduced)",
         "base13_rf": "[13] Alriyami 2026 RF (reproduced)"}


def run_report(cfg: dict) -> None:
    res, tdir = cfg["paths"]["results"], cfg["paths"]["results"] / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    rep = cfg["base_reported"]
    out = ["# Results\n", "Mean ± std over seeds " + str(cfg["seeds"]) +
           ". Thresholds for P/R/F1 are chosen on validation data. Base-paper 'reported' rows are copied "
           "from the papers; their splits/labels are not identical to ours, so the reproduced rows "
           "(same splits, same labels) are the fair comparison.\n"]

    def emit(name, title, t, note=""):
        t.to_csv(tdir / f"{name}.csv", index=False)
        out.append(f"\n## {title}\n\n{note}\n\n{_md(t)}\n")

    cert = _load(res / "raw")
    cert = cert[cert.get("method").notna()] if "method" in cert else pd.DataFrame()
    M = ["roc_auc", "pr_auc", "precision", "recall", "f1", "fpr"]
    if not cert.empty:
        cert["method_name"] = cert["method"].map(NAMES)
        # ---- T1 vs [11] ----
        d = cert[(cert.protocol == "chrono80") & (cert.variant == "full") & (cert.target == cert.release)]
        if not d.empty:
            t = _table(d, ["release", "method_name"], M)
            r11 = rep["b11_chrono80_r4.2"]
            t = pd.concat([pd.DataFrame([{"release": "r4.2", "method_name": "[11] Alkharabsheh 2026 (reported)",
                                          **{m: f"{r11[m]:.4f}" if m in r11 else "-" for m in M}}]), t])
            note = _sig(d, "ours", "base11", ["pr_auc", "f1", "roc_auc"])
            ours = d[d.method == "ours"]
            note += "\n\n" + "; ".join(f"{m}: ours {ours[m].mean():.4f} vs [11] reported {r11[m]:.4f} -> "
                                       f"{'higher' if ours[m].mean() > r11[m] else 'NOT higher'}" for m in r11)
            emit("T1_vs_base11", "Table 1 - CERT r4.2, chronological 80/20 (protocol of [11])", t, note)
        # ---- T2 vs [13] in-distribution ----
        d = cert[cert.protocol.isin(["chrono70", "userdisj70"]) & (cert.variant == "full") & (cert.target == cert.release)]
        if not d.empty:
            t = d.groupby(["release", "protocol", "method"])["pr_auc"].agg(_ms).unstack("method").reset_index()
            t["[13] reported (RF)"] = [f"{rep['b13_indist'].get(r, {}).get(p, np.nan):.4f}" for r, p in zip(t.release, t.protocol)]
            emit("T2_vs_base13_indist", "Table 2 - PR-AUC in-distribution (protocols of [13])", t)
        # ---- T3 cross-dataset ----
        d = cert[(cert.protocol == "userdisj70") & (cert.variant == "full")]
        if not d.empty:
            t = d.groupby(["release", "target", "method"])["pr_auc"].agg(_ms).unstack("method").reset_index()
            t = t.rename(columns={"release": "train (source)", "target": "test (target)"})
            t["[13] reported (RF)"] = [
                f"{rep['b13_indist'][s]['userdisj70']:.4f}" if s == g and s in rep["b13_indist"]
                else (f"{rep['b13_cross'].get(s, {}).get(g):.4f}" if rep["b13_cross"].get(s, {}).get(g) is not None else "-")
                for s, g in zip(t["train (source)"], t["test (target)"])]
            emit("T3_cross_dataset", "Table 3 - zero-shot cross-dataset PR-AUC (rows: source -> target)", t,
                 _gap(d, rep))
        # ---- T4 LOSO zero-day ----
        d = cert[cert.protocol == "loso"].copy()
        if not d.empty:
            d["scenario"] = d["scenario"].astype(int)
            d = d.sort_values(["release", "scenario", "method"])
            emit("T4_zero_day_loso", "Table 4 - zero-day: leave-one-insider-scenario-out",
                 _table(d, ["release", "scenario", "method_name"], ["pr_auc", "roc_auc", "recall", "f1", "fpr"]))
        # ---- T5 ablation ----
        e = cfg["experiments"]
        d = cert[(cert.protocol == e["ablation_protocol"]) & (cert.release == e["ablation_release"]) &
                 (cert.method == "ours") & (cert.target == cert.release)]
        if not d.empty:
            t = _table(d, ["variant"], ["pr_auc", "roc_auc", "f1", "fpr", "forward_rate"])
            full = d[d.variant == "full"].set_index("seed")["pr_auc"]
            t["delta_pr_auc_vs_full"] = [f"{(d[d.variant == v].set_index('seed')['pr_auc'] - full).mean():+.4f}"
                                         for v in t.variant]
            ge = ""
            a, b = d[d.variant == "cloud_only"].set_index("seed"), d[d.variant == "cloud_only_no_graph"].set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) >= 2:
                mean, lo, hi = paired_bootstrap(a.loc[common, "pr_auc"].values, b.loc[common, "pr_auc"].values)
                ge = (f"Graph effect on the cloud model (cloud_only vs cloud_only_no_graph): "
                      f"PR-AUC Δ = {mean:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
            emit("T5_ablation", f"Table 5 - ablation ({e['ablation_protocol']}, {e['ablation_release']})", t, ge)
            _figures(cfg, d)

    if not cert.empty and "final_val_pr_auc" in cert:
        d = cert[(cert.method == "ours") & (cert.variant == "full") & (cert.target == cert.release)]
        cols = [c for c in ("fog_train_pr_auc", "fog_val_pr_auc", "cloud_train_pr_auc_subsample",
                            "cloud_val_pr_auc", "final_val_pr_auc", "pr_auc") if c in d]
        emit("T10_fit_diagnostics", "Table 10 - over/under-fitting check (proposed): train vs validation vs test PR-AUC",
             _table(d, ["protocol", "release"], cols),
             "Large train >> val/test gaps would indicate over-fitting; low train scores would indicate under-fitting. "
             "cloud_train is measured on all training positives + a random set of training negatives (val-sized).")

    iot = _load(res / "raw" / "iot") if (res / "raw" / "iot").exists() else pd.DataFrame()
    if not iot.empty:
        d = iot[iot.dataset == "ton_iot"]
        if not d.empty:
            t = _table(d, ["protocol", "model"], ["accuracy", "f1_macro", "mcc", "roc_auc"])
            b = rep["b15_toniot_cv"]
            t = pd.concat([pd.DataFrame([{"protocol": "cv5", "model": "[15] Alamri 2026 XGBoost (reported)",
                                          **{k: f"{v:.4f}" for k, v in b.items()}, "roc_auc": "-"}]), t])
            emit("T6_ton_iot", "Table 6 - ToN-IoT device traffic (fog models vs [15])", t,
                 "Duplicates and identifier columns (IPs, timestamps) are removed before splitting.")
        d = iot[iot.dataset == "ciciomt2024"]
        if not d.empty:
            emit("T7_ciciomt2024", "Table 7 - CICIoMT2024 medical-device traffic (official train/test split)",
                 _table(d, ["protocol", "model"], ["accuracy", "f1_macro", "mcc", "roc_auc"]),
                 "device_insider = benign vs attacks a compromised registered device performs (spoofing, malformed/publish-flood MQTT).")

    sysf, chf = res / "bench" / "system.json", res / "chain" / "chain_results.json"
    if sysf.exists():
        sysj = json.loads(sysf.read_text())
        runs = pd.DataFrame(sysj["runs"])
        m = [c for c in ("forward_rate", "recall", "precision", "f1", "decision_ms_p50", "decision_ms_p95",
                         "alert_ms_p95", "uplink_kb", "backlog_max", "tau_mean", "fog_load_mean", "rtt_ms_measured",
                         "fog_cpu_pct", "fog_mem_mb_peak", "fog_energy_mwh_per_1k", "cloud_cpu_energy_mwh_per_1k",
                         "gpu_energy_mwh_per_1k", "total_energy_mwh_per_1k") if c in runs]
        runs[m] = runs[m].apply(pd.to_numeric, errors="coerce")
        t = _table(runs, ["scenario", "mode"], m)
        emit("T8a_fog_cloud_scenarios", "Table 8a - fog/cloud system under link and load conditions (RQ1, RQ4)", t,
             f"Fog nodes: {sysj['fog_mode']}; {sysj['n_events']:,} test events over {sysj['n_fog_nodes']} nodes; "
             f"mean ± std over {runs['repeat'].nunique()} repeats (conditions run in random order). "
             "decision = time until the event is decided (fog: locally; cloud_only: by the cloud); "
             "alert = time until a forwarded event is confirmed by the cloud. adaptive = forwarding threshold "
             "rises with measured fog load, link RTT and uplink backlog; static = fixed validation threshold; "
             "cloud_only = every record sent to the cloud. "
             f"CPU energy: {'RAPL counters, idle power subtracted' if sysj['rapl'] else 'estimated'}; "
             f"GPU energy: {'NVML counter' if sysj['nvml'] else 'not available'}.")
        b15 = rep["b15_system"]
        nrm = runs[runs.scenario == "normal"].groupby("mode")[m].mean()
        if "static" in nrm.index:
            g = lambda k, md="static": round(float(nrm.loc[md, k]), 4) if k in nrm and md in nrm.index else "-"
            emit("T8b_vs_base15", "Table 8b - normal conditions vs [15]", pd.DataFrame([
                ("fog decision latency p95 (ms)", g("decision_ms_p95"), "-"),
                ("fog batch of 50 (s, [15])", "-", b15["batch50_fog_s"]),
                ("fog CPU (% of 1 core)", g("fog_cpu_pct"), b15["cpu_pct"]),
                ("fog memory peak (MB)", g("fog_mem_mb_peak"), b15["mem_mb"]),
                ("uplink vs cloud-only (%)", round(100 * g("uplink_kb") / g("uplink_kb", "cloud_only"), 3)
                 if "cloud_only" in nrm.index else "-", "-"),
                ("total energy per 1k events (mWh), fog-cloud", g("total_energy_mwh_per_1k"), "-"),
                ("total energy per 1k events (mWh), cloud-only", g("total_energy_mwh_per_1k", "cloud_only"), "-"),
                ("energy per block (mWh, [15])", "-", b15["energy_mwh_per_block"]),
            ], columns=["metric", "proposed", "[15] Alamri 2026 (reported)"]))
    if chf.exists():
        c = json.loads(chf.read_text())
        o = c["operations"]
        rows = [(k, v["n"], round(v["gas_mean"]), round(v["latency_ms_mean"], 2), round(v["latency_ms_p95"], 2))
                for k, v in o.items()]
        t = pd.DataFrame(rows, columns=["operation", "n", "gas (mean)", "latency ms (mean)", "latency ms (p95)"])
        tests = c["security_tests"]
        note = (f"Alert anchoring latency {o['anchor_alert']['latency_ms_mean']:.2f} ms vs [15] block creation "
                f"{rep['b15_system']['block_ms']} ms. Gas per alert: single {c['gas_per_alert_single']:.0f}, "
                f"Merkle batch {c['gas_per_alert_batched']:.0f}. Tamper detection {c['tamper_detected']}/{c['tampered']}. "
                f"Security tests passed {sum(tests.values())}/{len(tests)}: "
                + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in tests.items())
                + f". Fog firmware attested = SHA-256 of the deployed fog model + runtime ({c.get('fog_firmware_sha256', '')[:16]}...).")
        emit("T8c_blockchain", "Table 8c - blockchain trust layer", t, note)
    shap = tdir / "shap_global.csv"
    if shap.exists():
        emit("T9_shap", "Table 9 - top fog-model features (mean |SHAP|)", pd.read_csv(shap).head(15))

    (res / "REPORT.md").write_text("\n".join(out))
    log.info(f"report: {res / 'REPORT.md'} and {len(list(tdir.glob('*.csv')))} CSV tables written")


def _sig(d, a, b, metrics) -> str:
    A, B = d[d.method == a].set_index("seed"), d[d.method == b].set_index("seed")
    common = A.index.intersection(B.index)
    if len(common) < 2:
        return ""
    parts = []
    for m in metrics:
        mean, lo, hi = paired_bootstrap(A.loc[common, m].values, B.loc[common, m].values)
        parts.append(f"{m}: Δ(ours − [11] reproduced) = {mean:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
    return "Paired bootstrap over seeds: " + "; ".join(parts)


def _gap(d, rep) -> str:
    msg = []
    for m in ("ours", "base13_rf"):
        x = d[d.method == m]
        ind = x[x.release == x.target]["pr_auc"].mean()
        cr = x[x.release != x.target]["pr_auc"].mean()
        msg.append(f"{NAMES[m]}: in-dist {ind:.4f}, cross {cr:.4f}, gap {ind - cr:.4f}")
    r = rep
    ind = np.mean([r["b13_indist"][k]["userdisj70"] for k in r["b13_indist"]])
    cr = np.mean([v for s in r["b13_cross"].values() for v in s.values()])
    msg.append(f"[13] reported (same CERT cells): in-dist {ind:.4f}, cross {cr:.4f}, gap {ind - cr:.4f}")
    return "Generalisation gap = mean in-distribution PR-AUC − mean cross-dataset PR-AUC.  \n" + "  \n".join(msg)


def _figures(cfg, abl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fdir = cfg["paths"]["results"] / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    e, s0 = cfg["experiments"], cfg["seeds"][0]
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in ("ours", "base11", "base13_rf"):
        f = cfg["paths"]["results"] / "scores" / f"{e['ablation_protocol']}__{e['ablation_release']}__{m}__full__seed{s0}.npz"
        if f.exists():
            z = np.load(f)
            p, r, _ = precision_recall_curve(z["y"], z["score"])
            ax.plot(r, p, label=NAMES[m])
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision"), ax.legend(fontsize=7)
    ax.set_title(f"PR curve - {e['ablation_release']} {e['ablation_protocol']} (seed {s0})")
    fig.tight_layout()
    fig.savefig(fdir / "pr_curves.png", dpi=200)
    plt.close(fig)
    g = abl.groupby("variant")["pr_auc"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(g.index, g["mean"], xerr=g["std"].fillna(0))
    ax.set_xlabel("PR-AUC"), ax.set_title("Ablation")
    fig.tight_layout()
    fig.savefig(fdir / "ablation.png", dpi=200)
    plt.close(fig)
