"""Every number, table and result figure of the paper, regenerated from the saved results.

    python reproduce.py paper        (or: from src.paper_outputs import run_paper_outputs)

Reads  results/raw/**.json (CERT + IoT runs), results/scores/*.npz, results/bench*/bench/system.json,
       results/chain/chain_results.json
Writes results/paper/PAPER_TABLES.md, one CSV per table, and results/paper/figures/{res_ablation,res_transfer}.pdf
Values are mean (± std) over the seeds in config.yaml; "reported" values are those published by the base studies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from .common import log

NAMES = {"ours": "ConFog", "base11": "HybridXAI (re-run)", "base13_rf": "RF (re-run)",
         "base13_rf_ctx": "RF (re-run) + context", "ours_v3": "ConFog v3 (not adopted)"}
VARIANTS = [("full", "ConFog (full)"), ("fog_only", "Fog tier only"), ("no_graph", "Without graph branch"),
            ("fixed_threshold", "Fixed alert threshold"), ("no_context", "Without context features"),
            ("no_tenure", "Without days of history"), ("no_web_keywords", "Without web keyword features"),
            ("fog_no_anomaly", "Without IF and AE scores"), ("cloud_only", "Cloud tier only (deep baseline)"),
            ("cloud_only_no_graph", "Cloud tier only, without graph"), ("drop_logon", "Without logon logs"),
            ("drop_file", "Without file logs"), ("drop_email", "Without e-mail logs"),
            ("drop_device", "Without removable-device logs"), ("drop_http", "Without web logs"),
            ("ae_only", "Autoencoder only"), ("if_only", "Isolation forest only")]


# ----------------------------------------------------------------------------- helpers
def _load(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(d.rglob("*.json")):
        x = json.loads(f.read_text())
        rows += x if isinstance(x, list) else [x]
    return pd.DataFrame(rows)


def _ms(x, nd=3) -> str:
    x = pd.Series(x).dropna().astype(float)
    if x.empty:
        return "--"
    return f"{x.mean():.{nd}f}" if len(x) == 1 else f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def _mean(x) -> float:
    x = pd.Series(x).dropna().astype(float)
    return float(x.mean()) if len(x) else np.nan


class _Out:
    def __init__(self, d: Path):
        self.d, self.md = d, ["# Paper tables (regenerated from results/)\n"]
        d.mkdir(parents=True, exist_ok=True)

    def table(self, name: str, title: str, t: pd.DataFrame, note: str = ""):
        t.to_csv(self.d / f"{name}.csv", index=False)
        cols = list(t.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
        self.md.append(f"\n## {title}\n\n" + (note + "\n\n" if note else "") + "\n".join(lines) + "\n")

    def text(self, title: str, body: str):
        self.md.append(f"\n## {title}\n\n{body}\n")

    def save(self):
        (self.d / "PAPER_TABLES.md").write_text("\n".join(self.md))


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _day_bootstrap(sdir: Path, proto, rel, a, b, B=2000, seed=0):
    """Paired bootstrap over test user-days (stratified by class), pooled over seeds."""
    rng, diffs, point = np.random.default_rng(seed), [], []
    for f in sorted(sdir.glob(f"{proto}__{rel}__{a}__full__seed*.npz")):
        g = Path(str(f).replace(f"__{a}__", f"__{b}__"))
        if not g.exists():
            continue
        A, Bz = np.load(f), np.load(g)
        y = A["y"]
        if len(y) != len(Bz["y"]) or (y != Bz["y"]).any():
            continue
        sa, sb = A["score"], Bz["score"]
        point.append(ap(y, sa) - ap(y, sb))
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for _ in range(B // 5):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            diffs.append(ap(y[i], sa[i]) - ap(y[i], sb[i]))
    if not point:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(point)), float(lo), float(hi), len(point)


# ----------------------------------------------------------------------------- CERT tables
def _cert_tables(cfg, c: pd.DataFrame, out: _Out, sdir: Path):
    rep = cfg["base_reported"]
    same = c.target == c.release
    full = c.variant == "full"

    # protocol of HybridXAI: chronological 80/20 on r4.2
    d = c[(c.protocol == "chrono80") & full & same & c.method.isin(["ours", "base13_rf", "base11"])]
    if len(d):
        r11 = rep["b11_chrono80_r4.2"]
        rows = [{"Model": "HybridXAI (reported)", "ROC-AUC": f"{r11['roc_auc']:.3f}", "PR-AUC": "NR",
                 "Precision": f"{r11['precision']:.2f}", "Recall": f"{r11['recall']:.2f}", "F1": f"{r11['f1']:.2f}",
                 "FPR": "NR"}]
        for m in ["base11", "base13_rf", "ours"]:
            x = d[d.method == m]
            rows.append({"Model": NAMES[m], "ROC-AUC": _ms(x.roc_auc, 5), "PR-AUC": _ms(x.pr_auc),
                         "Precision": _ms(x.precision), "Recall": _ms(x.recall), "F1": _ms(x.f1), "FPR": _ms(x.fpr, 5)})
        out.table("T_res11", "Protocol of HybridXAI (CERT r4.2, chronological 80/20)", pd.DataFrame(rows))

    # protocols of RF
    d = c[c.protocol.isin(["chrono70", "userdisj70"]) & full & same & c.release.isin(["r4.2", "r5.2"])]
    if len(d):
        rows = []
        for p, pn in (("chrono70", "Chronological 70/30"), ("userdisj70", "User-disjoint 70/30")):
            for r in ("r4.2", "r5.2"):
                x = d[(d.protocol == p) & (d.release == r)]
                rows.append({"Protocol": pn, "Release": r,
                             "RF (reported)": f"{rep['b13_indist'][r][p]:.3f}",
                             "RF (re-run)": _ms(x[x.method == "base13_rf"].pr_auc),
                             "RF (re-run) + context": _ms(x[x.method == "base13_rf_ctx"].pr_auc),
                             "ConFog": _ms(x[x.method == "ours"].pr_auc)})
        t = pd.DataFrame(rows)
        out.table("T_res13", "Protocols of RF (PR-AUC)", t)

    # transfer between releases
    d = c[(c.protocol == "userdisj70") & full & ~same]
    if len(d):
        rows = []
        for (s, g), x in d.groupby(["release", "target"]):
            rp = rep.get("b13_cross", {}).get(s, {}).get(g)
            rows.append({"Train -> test": f"{s} -> {g}", "RF (reported)": "--" if rp is None else f"{rp:.3f}",
                         "RF (re-run)": _ms(x[x.method == "base13_rf"].pr_auc),
                         "ConFog": _ms(x[x.method == "ours"].pr_auc)})
        out.table("T_transfer", "Transfer between releases (PR-AUC)", pd.DataFrame(rows))
        # generalisation gap = mean within-release (user-disjoint r4.2, r5.2) - mean cross-release
        within = c[(c.protocol == "userdisj70") & full & same & c.release.isin(["r4.2", "r5.2"])]
        gap = {}
        for m in ("base13_rf", "ours"):
            gap[NAMES[m]] = _mean(within[within.method == m].groupby("release").pr_auc.mean()) - \
                            _mean(d[d.method == m].groupby(["release", "target"]).pr_auc.mean())
        rw = np.mean([rep["b13_indist"][r]["userdisj70"] for r in ("r4.2", "r5.2")])
        rc = np.mean([v for s in rep["b13_cross"].values() for v in s.values()])
        gap["RF (reported)"] = rw - rc
        out.text("Generalisation gap (within-release user-disjoint mean minus cross-release mean, PR-AUC)",
                 "; ".join(f"{k}: {v:.3f}" for k, v in gap.items()))
        _fig_transfer(cfg, d, gap, rep)

    # unseen scenarios (LOSO), per scenario
    d = c[(c.protocol == "loso") & same].copy()
    if len(d):
        d["scenario"] = d["scenario"].astype(int)
        rows = []
        for (r, sc), x in d[d.variant == "full"].groupby(["release", "scenario"]):
            rows.append({"Release": r, "Scenario": sc,
                         "RF PR-AUC": _ms(x[x.method == "base13_rf"].pr_auc),
                         "ConFog PR-AUC": _ms(x[x.method == "ours"].pr_auc),
                         "RF recall": _ms(x[x.method == "base13_rf"].recall),
                         "ConFog recall": _ms(x[x.method == "ours"].recall)})
        out.table("T_loso", "Previously unseen insider scenarios (leave-one-scenario-out)", pd.DataFrame(rows))

    # ablation, chronological 80/20 r4.2 (figure + table)
    a = c[(c.protocol == "chrono80") & (c.release == "r4.2") & (c.method == "ours") & same]
    if len(a):
        rows = [{"Variant": n, "PR-AUC": _ms(a[a.variant == v].pr_auc), "F1": _ms(a[a.variant == v].f1),
                 "Forwarded": _ms(a[a.variant == v].forward_rate, 4)} for v, n in VARIANTS if (a.variant == v).any()]
        out.table("T_ablation_chrono80", "Ablation, CERT r4.2 chronological 80/20", pd.DataFrame(rows))
        _fig_ablation(cfg, a)

    # ablations and leakage checks under three protocols
    rows = []
    cols = [("chrono80", "r4.2"), ("userdisj70", "r4.2"), ("userdisj70", "r5.2"), ("loso", "r4.2"), ("loso", "r5.2")]
    entries = [("ours", v, n) for v, n in VARIANTS[:7]] + [("base13_rf", "full", "RF (re-run)"),
               ("base13_rf_ctx", "full", "RF (re-run) with context features"),
               ("ours", "cloud_only", "Deep sequence baseline (GRU + graph, all records)")]
    for m, v, n in entries:
        row = {"Variant": n}
        for p, r in cols:
            x = c[(c.protocol == p) & (c.release == r) & same & (c.method == m) & (c.variant == v)]
            if p == "loso":   # mean over scenarios of the per-scenario means
                row[f"{p} {r}"] = "--" if x.empty else f"{x.groupby('scenario').pr_auc.mean().mean():.3f}"
            else:
                row[f"{p} {r}"] = "--" if x.empty else f"{x.pr_auc.mean():.3f}"
        rows.append(row)
    out.table("T_ablation_protocols", "Ablations and leakage checks under three protocols (PR-AUC)", pd.DataFrame(rows))

    # overfitting check
    d = c[(c.method == "ours") & full & same]
    cols = [k for k in ("fog_train_pr_auc", "fog_val_pr_auc", "cloud_val_pr_auc", "pr_auc") if k in d]
    if len(d) and cols:
        t = d[d.protocol != "loso"].groupby(["protocol", "release"])[cols].agg(lambda s: _ms(s)).reset_index()
        out.table("T_fit", "Overfitting check (train / validation / test PR-AUC)", t)

    # blend weight w (stored in results of runs made after the review patch)
    if "blend_w" in c:
        d = c[(c.method == "ours") & full & same & c.blend_w.notna()]
        if len(d):
            t = d.groupby(["protocol", "release"]).blend_w.agg(["mean", "count", lambda s: int((s > 0).sum())])
            t.columns = ["mean w", "runs", "runs with w > 0"]
            out.table("T_blend_w", "Blend weight w chosen on validation", t.round(2).reset_index())

    # paired bootstrap over test user-days
    rows = []
    for p, r in (("chrono80", "r4.2"), ("chrono70", "r4.2"), ("chrono70", "r5.2"), ("userdisj70", "r4.2"), ("userdisj70", "r5.2")):
        for b in ("base13_rf", "base11"):
            res = _day_bootstrap(sdir, p, r, "ours", b)
            if res:
                rows.append({"Protocol": p, "Release": r, "Comparison": f"ConFog - {NAMES[b]}",
                             "dPR-AUC": f"{res[0]:+.4f}", "95% CI": f"[{res[1]:+.4f}, {res[2]:+.4f}]", "Seeds": res[3]})
    if rows:
        out.table("T_bootstrap", "Paired bootstrap over test user-days", pd.DataFrame(rows))


# ----------------------------------------------------------------------------- IoT, testbed, ledger
def _iot_table(cfg, iot: pd.DataFrame, out: _Out):
    if iot.empty:
        return
    t = iot.groupby(["dataset", "protocol", "model"])[["f1_macro", "mcc"]].agg(lambda s: _ms(s)).reset_index()
    out.table("T_iot", "Medical-device traffic (macro F1, MCC)", t,
              f"FogXGB reported on ToN-IoT 5-fold CV: {cfg['base_reported']['b15_toniot_cv']}")


def _bench_tables(res: Path, out: _Out):
    for name, title in (("bench", "Testbed, 10,000-day sample"), ("bench_full", "Testbed, full test set"),
                        ("bench_stress_default", "Saturated gateway, default constants"),
                        ("bench_stress_aggressive", "Saturated gateway, stricter constants"),
                        ("bench_stress_lenient", "Saturated gateway, more lenient constants"),
                        ("bench_full_rapl", "Testbed energy, full test set (RAPL, idle power measured before each run)"),
                        ("bench_realistic", "Testbed at realistic load (20 records/s per node, 6000 user-days)")):  # PATCH_V13
        f = (res / name / "bench" / "system.json") if name != "bench" else (res / "bench" / "system.json")
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        r = pd.DataFrame(j["runs"])
        cols = [k for k in ("forward_rate", "recall", "decision_ms_p50", "decision_ms_p95", "alert_ms_p95", "uplink_kb",
                            "fog_cpu_pct", "tau_mean", "total_energy_mwh_per_1k") if k in r]
        if name != "bench_full_rapl":   # energy is reported only from the run with the corrected meter
            cols = [k for k in cols if k != "total_energy_mwh_per_1k"]
        t = r.groupby(["scenario", "mode"], sort=False)[cols].agg(lambda s: _ms(s, 3)).reset_index()
        note = f"{j['n_events']:,} user-days, {j['n_fog_nodes']} fog nodes, {j['fog_mode']}, {r.repeat.nunique()} repeats."
        if "malicious" in r:
            k = r[(r["mode"] == "static") & (r.scenario.isin(["normal", "saturated"]))]
            if len(k):
                n = int(k.malicious.iloc[0])
                hit = int(round(k.recall.mean() * n))
                lo, hi = _wilson(hit, n)
                note += f" Static mode: recall {hit}/{n}, Wilson 95% interval {lo:.2f}-{hi:.2f}."
        out.table(f"T_{name}", title, t, note)


def _chain_table(res: Path, out: _Out):
    for d, name, title in (("chain", "T_chain", "Ledger operations, single-node Hardhat"),
                           ("chain_besu", "T_chain_besu", "Ledger operations, four-validator Hyperledger Besu (QBFT)")):
        _chain_one(res / d / "chain" / "chain_results.json" if d != "chain" else res / "chain" / "chain_results.json",
                   name, title, out)


def _review_tables(res: Path, out: _Out):
    d = res / "review"
    if (d / "tradeoff_choice.json").exists():
        j = json.loads((d / "tradeoff_choice.json").read_text())
        rows = []
        for proto, v in j.items():
            for lab, c in (("chosen on validation", v["chosen"]), ("paper default", v["paper_default"])):
                if c:
                    rows.append({"Protocol": proto, "Configuration": lab, "Leaves": int(c["num_leaves"]),
                                 "Anomaly scores": bool(c["anomaly_scores"]), "Size (KB)": f"{c['kb']:.0f}",
                                 "Latency (ms/record)": f"{c['ms']:.3f}", "Validation PR-AUC": f"{c['val']:.3f}",
                                 "Test PR-AUC": f"{c['test']:.3f} ± {c['test_sd']:.3f}"})
        out.table("T_tradeoff", "Fog model size, latency and accuracy", pd.DataFrame(rows),
                  "Rule fixed in advance: " + next(iter(j.values()))["rule"] + ". Mean over three seeds.")
    if (d / "mimicry_mean.csv").exists():
        t = pd.read_csv(d / "mimicry_mean.csv")
        t = pd.DataFrame({"Model": t.model, "alpha": t.alpha,
                          "Recall": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.recall, t.recall_sd)],
                          "PR-AUC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.pr_auc, t.pr_auc_sd)]})
        out.table("T_mimicry", "Insiders who imitate their normal behaviour (user-disjoint r4.2, five seeds)", t)
    if (d / "faithfulness.json").exists():
        j = json.loads((d / "faithfulness.json").read_text())
        rows = [{"Inputs replaced": k[3:], "Score drop, reasons": f"{v['mean_drop_reasons']:.3f}",
                 "Score drop, random": f"{v['mean_drop_random']:.3f}", "Reasons larger": f"{v['share_reasons_larger']:.1%}",
                 "No longer forwarded (reasons / random)":
                     f"{v['share_no_longer_forwarded_reasons']:.1%} / {v['share_no_longer_forwarded_random']:.1%}",
                 "Wilcoxon p": f"{v['wilcoxon_p']:.1e}"} for k, v in j.items() if k.startswith("top")]
        out.table("T_faithfulness", f"Faithfulness of the alert reasons ({j['n_alerts']} forwarded days)", pd.DataFrame(rows))


def _chain_one(f: Path, name: str, title: str, out: _Out):
    if not f.exists():
        return
    j = json.loads(f.read_text())
    t = pd.DataFrame([{"Operation": k, "Calls": v["n"], "Gas (mean)": f"{v['gas_mean']:,.0f}",
                       "Latency mean (ms)": f"{v['latency_ms_mean']:.2f}", "Latency p95 (ms)": f"{v['latency_ms_p95']:.2f}"}
                      for k, v in j["operations"].items()])
    out.table(name, title, t,
              f"Node: {j.get('node')}; tamper detected {j['tamper_detected']}/{j['tampered']}; "
              f"security tests: {j['security_tests']}")


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import glob as _g, matplotlib.font_manager as _fm
    for _f in (_g.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf")
               + _g.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyrepagella-*.otf")
               + _g.glob("/usr/share/fonts/**/texgyrepagella-*.otf", recursive=True)
               + _g.glob(str(Path.home() / ".fonts/texgyrepagella-*.otf"))):
        _fm.fontManager.addfont(_f)
    _SERIF = ["TeX Gyre Pagella", "Palatino Linotype", "Palatino", "URW Palladio L", "DejaVu Serif"]
    plt.rcParams.update({"font.family": "serif", "font.serif": _SERIF, "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "legend.fontsize": 8.5, "hatch.color": "white"})
    return plt


C_PROP, C_PUB, C_RF, C_LIGHT = "#0072B2", "#D55E00", "#CC79A7", "#8fb8d9"


def _save(plt, fig, d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(d / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _fig_ablation(cfg, a: pd.DataFrame):
    plt = _plt()
    rows = [(n, a[a.variant == v]) for v, n in VARIANTS if (a.variant == v).any()
            and v not in ("no_tenure", "no_web_keywords")]
    full = [r for r in rows if r[0] == "ConFog (full)"]   # reference bar always on top
    rows = sorted([r for r in rows if r[0] != "ConFog (full)"], key=lambda r: r[1].pr_auc.mean()) + full
    y = np.arange(len(rows))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True)
    for ax, m, lab in zip(axs, ("pr_auc", "f1"), ("PR-AUC\n(a)", "F1 score\n(b)")):
        mu = np.array([r[1][m].mean() for r in rows])
        sd = np.array([r[1][m].std(ddof=1) for r in rows])
        ax.barh(y, mu, xerr=sd, color=[C_PROP if r[0] == "ConFog (full)" else C_LIGHT for r in rows],
                error_kw=dict(lw=0.8, capsize=2))
        for i, (mm, ss) in enumerate(zip(mu, sd)):
            ax.text(min(mm + ss + 0.015, 1.03), i, f"{mm:.3f}", va="center", fontsize=6.5, color="#333333")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [r[0] for r in rows], fontsize=8)
    fig.tight_layout()
    _save(plt, fig, cfg["paths"]["results"] / "paper" / "figures", "res_ablation")


def _fig_transfer(cfg, d: pd.DataFrame, gap: dict, rep: dict):
    plt = _plt()
    pairs = [("r4.2", "r5.2"), ("r5.2", "r4.2"), ("r6.2", "r4.2"), ("r6.2", "r5.2")]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [2.8, 1]})
    x, w = np.arange(len(pairs)), 0.8 / 3
    series = [("RF (reported)", C_PUB, True, [rep["b13_cross"].get(s, {}).get(g, np.nan) for s, g in pairs], None)]
    for m, lab, col in (("base13_rf", "RF (re-run)", C_RF), ("ours", "ConFog", C_PROP)):
        xs = [d[(d.method == m) & (d.release == s) & (d.target == g)].pr_auc for s, g in pairs]
        series.append((lab, col, False, [s.mean() for s in xs], [s.std(ddof=1) for s in xs]))
    for i, (lab, col, hatched, mu, sd) in enumerate(series):
        kw = dict(hatch="///", edgecolor="white", linewidth=0) if hatched else {}
        b = axs[0].bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=2, color=col, label=lab, error_kw=dict(lw=0.8), **kw)
        for bb, m, s in zip(b, mu, sd or [0] * len(mu)):
            if np.isfinite(m):
                axs[0].text(bb.get_x() + bb.get_width() / 2, m + (s or 0) + 0.012, f"{m:.2f}", ha="center",
                            va="bottom", fontsize=7.5, color="#333333")
    axs[0].set_xticks(x, [f"{s}→{g}" for s, g in pairs], fontsize=8)
    axs[0].set_ylabel("PR-AUC")
    axs[0].set_ylim(0, 1.12)
    axs[0].set_xlabel("Training → test release\n(a)")
    keys = ["RF (reported)", "RF (re-run)", "ConFog"]
    vals = [gap[k] for k in keys]
    b = axs[1].bar([0, 1, 2], vals, 0.6, color=[C_PUB, C_RF, C_PROP])
    b[0].set_hatch("///")
    b[0].set_edgecolor("white")
    for bb, v in zip(b, vals):
        axs[1].text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    axs[1].set_xticks([0, 1, 2], ["RF\n(reported)", "RF\n(re-run)", "ConFog"], fontsize=8)
    axs[1].set_ylabel("Generalisation gap (PR-AUC)")
    axs[1].set_ylim(0, 0.5)
    axs[1].set_xlabel("(b)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.68, -0.33), ncol=3)
    fig.tight_layout()
    _save(plt, fig, cfg["paths"]["results"] / "paper" / "figures", "res_transfer")


# ----------------------------------------------------------------------------- entry point
def run_paper_outputs(cfg: dict) -> None:
    res = cfg["paths"]["results"]
    out = _Out(res / "paper")
    allraw = _load(res / "raw")
    cert = allraw[allraw["method"].notna()] if "method" in allraw else pd.DataFrame()
    iot = _load(res / "raw" / "iot") if (res / "raw" / "iot").exists() else pd.DataFrame()
    if len(cert):
        _cert_tables(cfg, cert, out, res / "scores")
    _iot_table(cfg, iot, out)
    _bench_tables(res, out)
    _chain_table(res, out)
    _review_tables(res, out)
    out.save()
    log.info(f"paper outputs: {res / 'paper' / 'PAPER_TABLES.md'} and {res / 'paper' / 'figures'}")

# PATCH_V11
