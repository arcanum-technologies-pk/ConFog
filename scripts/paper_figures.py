"""Publication figures (MDPI style): no titles, axis labels only, 600-dpi PNG + vector PDF.

Model names in all figures (one word each; the citation goes in the caption and the text, never on the image)
  ConFog            - proposed context-aware fog-cloud framework
  HybridXAI         - model of [11];  "(reported)" = value published in [11] (hatched bar),
                      "(re-run)" = re-implemented and trained on our features and splits
  RF                - random-forest configuration of [13]; same "(reported)" / "(re-run)" convention
  FogXGB            - fog-cloud XGBoost system of [15]; same convention
Multi-panel figures carry (a), (b), ... under each panel; the meaning goes in the caption.

usage: python paper_figures.py [results_folder]   (default: results/; writes <folder>/paper_figures/)
Independent of config.yaml: the published numbers of [11], [13], [15] are fixed constants below.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results").resolve()
if not (RES / "raw").is_dir():
    sys.exit(f"{RES}/raw not found - pass the results folder, e.g. python paper_figures.py results")
OUT = RES / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
# values reported in the published papers (not re-computed)
REP = {
    "b11_chrono80_r4.2": {"roc_auc": 0.976, "precision": 0.89, "recall": 0.86, "f1": 0.87},
    "b13_indist": {"r4.2": {"chrono70": 0.6660, "userdisj70": 0.5497}, "r5.2": {"chrono70": 0.8150, "userdisj70": 0.6254}},
    "b13_cross": {"r4.2": {"r5.2": 0.250}, "r5.2": {"r4.2": 0.390}, "r6.2": {"r4.2": 0.015, "r5.2": 0.135}},
    "b15_toniot_cv": {"accuracy": 1.000, "f1_macro": 1.000, "mcc": 1.000},
    "b15_system": {"block_ms": 10.11},
}

# colour-blind-safe roles (validated): proposed / published / baselines
C_PROP, C_PUB, C_RF, C_ENS = "#0072B2", "#D55E00", "#CC79A7", "#009E73"
MODE_C = {"adaptive": C_PROP, "static": C_RF, "cloud_only": C_PUB}
MODE_N = {"adaptive": "Adaptive forwarding", "static": "Static forwarding", "cloud_only": "Cloud-only"}
SCEN_N = {"normal": "Normal", "congested_link": "Congested\nlink", "overloaded_fog": "Overloaded\nfog",
          "degraded": "Degraded"}
PROP = "ConFog"                       # proposed model name: change here only
N11, N13, N15 = "HybridXAI", "RF", "FogXGB"
METH_N = {"ours": PROP, "base13_rf": f"{N13} (re-run)", "base11": f"{N11} (re-run)"}
METH_C = {"ours": C_PROP, "base13_rf": C_RF, "base11": C_ENS}
PUB = dict(hatch="///", edgecolor="white", linewidth=0)

from pathlib import Path
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
                     "legend.frameon": False, "legend.fontsize": 8.5, "savefig.bbox": "tight",
                     "hatch.color": "white", "hatch.linewidth": 0.8})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", name)


def load_rows(sub):
    rows = []
    for f in (RES / "raw").rglob("*.json"):
        if ("iot" in f.parts) == (sub == "iot"):
            d = json.loads(f.read_text())
            rows += d if isinstance(d, list) else [d]
    return pd.DataFrame(rows)


def mstd(df, col):
    return df[col].mean(), (df[col].std(ddof=1) if len(df) > 1 else 0.0)


def bar_labels(ax, bars, vals, fmt="{:.3f}", dy=0.012, size=7):
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                    ha="center", va="bottom", fontsize=size, color="#333333")


cert = load_rows("cert")
iot = load_rows("iot")
own = cert[cert.target == cert.release] if not cert.empty else cert
print(f"{len(cert)} CERT rows, {len(iot)} IoT rows -> {OUT}")

# ---------------------------------------------------------------- Fig 1: vs published [11]
d = own[(own.protocol == "chrono80") & (own.method == "ours") & (own.variant == "full")]
if not d.empty:
    m = ["roc_auc", "precision", "recall", "f1"]
    lab = ["ROC-AUC", "Precision", "Recall", "F1-score"]
    pub = [REP["b11_chrono80_r4.2"][k] for k in m]
    ours = [mstd(d, k) for k in m]
    x, w = np.arange(len(m)), 0.36
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    b1 = ax.bar(x - w / 2, pub, w, color=C_PUB, label=f"{N11} (reported)", **PUB)
    b2 = ax.bar(x + w / 2, [o[0] for o in ours], w, yerr=[o[1] for o in ours], capsize=2,
                color=C_PROP, label=PROP, error_kw=dict(lw=0.8))
    bar_labels(ax, b1, pub)
    bar_labels(ax, b2, [o[0] for o in ours], dy=0.025)
    ax.set_xticks(x, lab)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)        # bars start at zero (no truncated axis)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig01_vs_published_11")

# ---------------------------------------------------------------- Fig 2: vs published [13]
cells = [("r4.2", "r4.2", "chrono70", "r4.2\ntemporal"), ("r4.2", "r4.2", "userdisj70", "r4.2\nunseen\nusers"),
         ("r5.2", "r5.2", "chrono70", "r5.2\ntemporal"), ("r5.2", "r5.2", "userdisj70", "r5.2\nunseen\nusers"),
         ("r4.2", "r5.2", "userdisj70", "r4.2→r5.2"), ("r5.2", "r4.2", "userdisj70", "r5.2→r4.2"),
         ("r6.2", "r4.2", "userdisj70", "r6.2→r4.2"), ("r6.2", "r5.2", "userdisj70", "r6.2→r5.2")]
o = cert[(cert.method == "ours") & (cert.variant == "full")] if not cert.empty else cert
if not o.empty:
    pub, mean, sd, labs = [], [], [], []
    for src, tgt, proto, lab in cells:
        pub.append(REP["b13_indist"][src][proto] if src == tgt else REP["b13_cross"].get(src, {}).get(tgt, np.nan))
        s = o[(o.release == src) & (o.target == tgt) & (o.protocol == proto)]
        mean.append(s.pr_auc.mean() if len(s) else np.nan)
        sd.append(s.pr_auc.std(ddof=1) if len(s) > 1 else 0)
        labs.append(lab)
    x, w = np.arange(len(cells)), 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    b1 = ax.bar(x - w / 2, pub, w, color=C_PUB, label=f"{N13} (reported)", **PUB)
    b2 = ax.bar(x + w / 2, mean, w, yerr=sd, capsize=2, color=C_PROP, label=PROP, error_kw=dict(lw=0.8))
    for bars, vals, errs in ((b1, pub, [0] * len(pub)), (b2, mean, sd)):
        for b, v, e in zip(bars, vals, errs):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + e + 0.015, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=90, color="#333333")
    ax.set_xticks(x, labs, fontsize=7.5)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Training → test release / protocol")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2)
    save(fig, "fig02_vs_published_13")

# ---------------------------------------------------------------- Fig 3: baselines on identical splits
sets = [("chrono80", "r4.2", "r4.2\n80/20\ntemporal"), ("chrono70", "r4.2", "r4.2\n70/30\ntemporal"),
        ("chrono70", "r5.2", "r5.2\n70/30\ntemporal"), ("userdisj70", "r4.2", "r4.2\nunseen\nusers"),
        ("userdisj70", "r5.2", "r5.2\nunseen\nusers")]
if not own.empty:
    meths = ["ours", "base13_rf", "base11"]
    x, w = np.arange(len(sets)), 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, mth in enumerate(meths):
        vals = [own[(own.protocol == p) & (own.release == r) & (own.method == mth) & (own.variant == "full")].pr_auc
                for p, r, _ in sets]
        mu = [v.mean() if len(v) else np.nan for v in vals]
        sd = [v.std(ddof=1) if len(v) > 1 else 0 for v in vals]
        b = ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=2, color=METH_C[mth], label=METH_N[mth],
                   error_kw=dict(lw=0.8))
        bar_labels(ax, b, np.array(mu) + np.nan_to_num(sd), fmt="", size=6)
        for bb, v, e in zip(b, mu, sd):
            if np.isfinite(v):
                ax.text(bb.get_x() + bb.get_width() / 2, v + e + 0.015, f"{v:.3f}", ha="center",
                        fontsize=6, rotation=90, va="bottom", color="#333333")
    ax.set_xticks(x, [s[2] for s in sets], fontsize=7.5)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=3)
    save(fig, "fig03_baselines_identical_splits")

# ---------------------------------------------------------------- Fig 4: ablation
VAR_N = {"full": f"{PROP} (full)", "no_graph": "w/o graph branch", "cloud_only_no_graph": "Cloud tier only, w/o graph",
         "no_context": "w/o context features", "fixed_threshold": "Fixed threshold",
         "cloud_only": "Cloud tier only", "fog_only": "Fog tier only", "fog_no_anomaly": "w/o IF/AE scores",
         "if_only": "Isolation forest only", "ae_only": "Autoencoder only", "drop_logon": "w/o logon logs",
         "drop_device": "w/o removable-device logs", "drop_file": "w/o file logs",
         "drop_email": "w/o e-mail logs", "drop_http": "w/o web logs"}
e = {"ablation_protocol": "chrono80", "ablation_release": "r4.2"}
a = own[(own.protocol == e["ablation_protocol"]) & (own.release == e["ablation_release"]) & (own.method == "ours")]
if not a.empty:
    g = a.groupby("variant").pr_auc.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    col = [C_PROP if v == "full" else "#8fb8d9" for v in g.index]
    ax.barh([VAR_N.get(v, v) for v in g.index], g["mean"], xerr=g["std"].fillna(0), color=col,
            error_kw=dict(lw=0.8, capsize=2))
    for i, (mu, sd) in enumerate(zip(g["mean"], g["std"].fillna(0))):
        ax.text(min(mu + sd + 0.01, 1.02), i, f"{mu:.3f}", va="center", fontsize=7, color="#333333")
    ax.set_xlabel("PR-AUC")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    save(fig, "fig04_ablation")

# ---------------------------------------------------------------- Fig 5: cross-dataset heatmaps
if not o.empty:
    src, tgt = ["r4.2", "r5.2", "r6.2"], ["r4.2", "r5.2"]

    def grid(method=None, published=False):
        M = np.full((3, 2), np.nan)
        for i, s in enumerate(src):
            for j, t in enumerate(tgt):
                if published:
                    M[i, j] = REP["b13_indist"][s]["userdisj70"] if s == t else REP["b13_cross"].get(s, {}).get(t, np.nan)
                else:
                    q = cert[(cert.method == method) & (cert.variant == "full") & (cert.protocol == "userdisj70")
                             & (cert.release == s) & (cert.target == t)]
                    M[i, j] = q.pr_auc.mean() if len(q) else np.nan
        return M
    panels = [(grid("ours"), f"(a) {PROP}"), (grid("base13_rf"), f"(b) {N13} (re-run)"),
              (grid(published=True), f"(c) {N13} (reported)")]
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 3.0), sharey=True)
    for ax, (M, lab) in zip(axs, panels):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(2):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 else "#222222")
        ax.set_xticks(range(2), tgt)
        ax.set_yticks(range(3), src)
        ax.set_xlabel(f"Test release\n{lab}")
        ax.grid(False)
    axs[0].set_ylabel("Training release")
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="PR-AUC")
    save(fig, "fig05_cross_dataset")

# ---------------------------------------------------------------- Fig 6: zero-day heatmap
z = cert[(cert.protocol == "loso") & (cert.variant == "full")] if not cert.empty else cert
if not z.empty:
    z = z.assign(scenario=z.scenario.astype(int))
    piv = z.groupby(["release", "scenario", "method"]).pr_auc.mean().unstack("method")[["ours", "base13_rf", "base11"]]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "#222222")
    ax.set_xticks(range(3), [PROP, f"{N13}\n(re-run)", f"{N11}\n(re-run)"], fontsize=8)
    ax.set_yticks(range(len(piv)), [f"{r}, scenario {s}" for r, s in piv.index])
    ax.set_ylabel("Held-out insider scenario")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="PR-AUC")
    save(fig, "fig06_zero_day")

# ---------------------------------------------------------------- Fig 7: confusion matrices (integer counts)
# Each seed gives exact integer counts on the same chronological test set:
#   TP = recall x positives, FP = FPR x negatives. The seed with the median F1 of each model is shown,
#   so every cell is a whole number of user-days (never an average).
d = own[(own.protocol == "chrono80") & (own.variant == "full")]
if not d.empty:
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    for k, (ax, mth) in enumerate(zip(axs, ["ours", "base13_rf", "base11"])):
        s = d[d.method == mth].sort_values("f1").reset_index(drop=True)
        r = s.iloc[(len(s) - 1) // 2]                       # median-F1 seed
        npos, nneg = int(r.n_pos), int(r.n - r.n_pos)
        tp, fp = int(round(r.recall * npos)), int(round(r.fpr * nneg))
        M = np.array([[nneg - fp, fp], [npos - tp, tp]], dtype=int)
        R = M / M.sum(1, keepdims=True)
        print(f"  confusion {mth}: seed {int(r.seed)}  TN={M[0,0]} FP={fp} FN={M[1,0]} TP={tp}")
        ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                pc = 100 * R[i, j]
                pct = ("<0.01%" if 0 < pc < 0.01 else (f"{pc:.3f}%" if 99.99 < pc < 100 else f"{pc:.2f}%"))
                ax.text(j, i, f"{M[i, j]:,d}\n({pct})", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] > 0.6 else "#222222")
        ax.set_xticks([0, 1], ["Normal", "Malicious"])
        ax.set_yticks([0, 1], ["Normal", "Malicious"])
        ax.set_xlabel(f"Predicted label\n({'abc'[k]}) {METH_N[mth]}")
        ax.grid(False)
    axs[0].set_ylabel("True label")
    save(fig, "fig07_confusion_matrices")

# ---------------------------------------------------------------- Fig 8: PR and ROC curves (first seed)
seed = int(sorted(cert["seed"].unique())[0])
curves = {m: RES / "scores" / f"chrono80__r4.2__{m}__full__seed{seed}.npz" for m in ("ours", "base13_rf", "base11")}
if all(p.exists() for p in curves.values()):
    ls = {"ours": "-", "base13_rf": "--", "base11": ":"}
    for kind in ("pr", "roc"):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        for m, p in curves.items():
            zz = np.load(p)
            if kind == "pr":
                pr, rc, _ = precision_recall_curve(zz["y"], zz["score"])
                ax.plot(rc, pr, ls[m], color=METH_C[m], lw=1.6, label=METH_N[m])
            else:
                fp, tp, _ = roc_curve(zz["y"], zz["score"])
                ax.plot(np.maximum(fp, 1e-5), tp, ls[m], color=METH_C[m], lw=1.6, label=METH_N[m])
        if kind == "pr":
            ax.set_xlabel("Recall\n(a)")
            ax.set_ylabel("Precision")
        else:
            ax.set_xscale("log")
            ax.set_xlim(1e-5, 1)
            ax.set_xlabel("False-positive rate (log scale)\n(b)")
            ax.set_ylabel("True-positive rate")
        ax.set_ylim(-0.02, 1.03)
        ax.legend(loc="center right" if kind == "pr" else "center left", fontsize=7.5)
        if kind == "pr":   # zoomed inset: ConFog and RF overlap near the top-right corner
            axin = ax.inset_axes([0.19, 0.28, 0.40, 0.40])   # PATCH_V12
            for m, p in curves.items():
                if m == "base11":
                    continue
                zz = np.load(p)
                pr, rc, _ = precision_recall_curve(zz["y"], zz["score"])
                axin.plot(rc, pr, ls[m], color=METH_C[m], lw=1.3)
            axin.set_xlim(0.90, 1.0)
            axin.set_ylim(0.80, 1.01)
            axin.set_xticks([0.90, 0.95, 1.00])
            axin.set_yticks([0.90, 1.00])
            axin.tick_params(labelsize=6.5)
            for sp in axin.spines.values():
                sp.set_visible(True)
                sp.set_color("#888888")
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((0.90, 0.80), 0.10, 0.21, fill=False, edgecolor="#888888", lw=0.9))
            ax.legend(loc="upper center", bbox_to_anchor=(0.55, 0.93), fontsize=7.5)   # PATCH_V14
        save(fig, f"fig08{'a' if kind == 'pr' else 'b'}_{kind}_curve")

# ---------------------------------------------------------------- Fig 9: score distribution
p = curves.get("ours")
if p is not None and p.exists():
    zz = np.load(p)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    bins = np.linspace(0, 2, 81)
    ax.hist(zz["score"][zz["y"] == 0], bins, color="#9a9a9a", label="Normal user-days")
    ax.hist(zz["score"][zz["y"] == 1], bins, color=C_PROP, label="Malicious user-days")
    ax.axvline(1.0, color="#555555", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Final score S")
    ax.set_ylabel("Number of user-days (log scale)")
    ax.legend(loc="upper right")
    save(fig, "fig09_score_distribution")

# ---------------------------------------------------------------- Fig 10: SHAP importance (readable names)
FEAT = {"http": "web visits", "email": "e-mails sent", "file": "file copies", "usb_conn": "USB connections",
        "logon": "logons", "new_pc": "new PCs used", "http_job": "job-site visits", "http_leak": "upload/leak-site visits",
        "http_hack": "hacking-site visits", "file_zip": "archive files", "file_exe": "executable files",
        "file_doc": "document files", "email_ext": "external e-mails", "email_bcc": "BCC e-mails",
        "email_att": "attachments", "email_size_sum": "e-mail volume", "pc": "PCs used", "logon_off": "off-hour logons",
        "usb_off": "off-hour USB use", "file_off": "off-hour file copies", "email_off": "off-hour e-mails",
        "http_off": "off-hour web visits", "logoff": "logoffs"}
SPECIAL = {"ctx_tenure": "Days of history (context)", "if_score": "Isolation-forest score", "ae_score": "Autoencoder score",
           "first_hour": "First activity hour", "last_hour": "Last activity hour", "span_hours": "Active hours span",
           "dev_first_hour": "Deviation in first activity hour", "dev_last_hour": "Deviation in last activity hour",
           "off_ratio": "Off-hour activity ratio", "ctx_new_device": "New device (context)",
           "ctx_offhours": "Off-hours (context)", "ctx_night_shift": "Night shift (context)",
           "ctx_shift_offset": "Shift offset (context)", "ctx_weekend": "Weekend (context)"}


def nice(f):
    if f in SPECIAL:
        return SPECIAL[f]
    if f.startswith("ctx_role_"):
        return f"Role: {f[9:].replace('_', ' ').replace('it admin', 'IT admin')} (context)"
    dev = f.startswith("dev_")
    base = f[4:] if dev else f
    base = base[:-2] if base.endswith("_n") else base
    name = FEAT.get(base, base.replace("_", " "))
    return (f"Deviation in {name}" if dev else name[0].upper() + name[1:])


shap = RES / "tables" / "shap_global.csv"
if shap.exists():
    s = pd.read_csv(shap).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.barh([nice(f) for f in s.feature], s.mean_abs_shap, color=C_PROP)
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_shap_importance")

# ---------------------------------------------------------------- Fig 11-12: system benchmark
sysf = RES / "bench" / "system.json"
if sysf.exists():
    runs = pd.DataFrame(json.loads(sysf.read_text())["runs"])
    sc = [s for s in SCEN_N if s in set(runs.scenario)]
    modes = [m for m in MODE_N if m in set(runs["mode"])]
    x = np.arange(len(sc))

    def grouped(col, ylab, name, log=False, fmt="{:.0f}"):
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        w = 0.8 / len(modes)
        for i, md in enumerate(modes):
            g = runs[runs["mode"] == md].groupby("scenario")[col]
            mu, sd = g.mean().reindex(sc), g.std(ddof=1).reindex(sc).fillna(0)
            b = ax.bar(x + (i - (len(modes) - 1) / 2) * w, mu, w, yerr=sd, capsize=2, color=MODE_C[md],
                       label=MODE_N[md], error_kw=dict(lw=0.8))
            for bb, v, e in zip(b, mu, sd):
                y = (v + e) * 1.12 if log else v + e + 0.04
                ax.text(bb.get_x() + bb.get_width() / 2, y, fmt.format(v), ha="center", va="bottom", fontsize=6,
                        rotation=90, color="#333333")
        ax.margins(y=0.18)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x, [SCEN_N[s] for s in sc])
        ax.set_ylabel(ylab)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
        save(fig, name)
    grouped("decision_ms_p95", "p95 decision latency (ms, log scale)", "fig11a_decision_latency", log=True,
            fmt="{:,.1f}")
    grouped("uplink_kb", "Uplink data (KB, log scale)", "fig11b_uplink", log=True, fmt="{:,.1f}")
    grouped("total_energy_mwh_per_1k", "Energy per 1000 events (mWh)", "fig11c_energy", fmt="{:.2f}")

chf = RES / "chain" / "chain_results.json"
if chf.exists():
    c = json.loads(chf.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    v = [c["gas_per_alert_single"], c["gas_per_alert_batched"]]
    b = axs[0].bar(["Per-alert\nanchoring", "Merkle-batch\nanchoring"], v, color=[C_PROP, "#8fb8d9"], width=0.55)
    for bb, vv in zip(b, v):
        axs[0].text(bb.get_x() + bb.get_width() / 2, vv * 1.02, f"{vv:,.0f}", ha="center", va="bottom", fontsize=7.5)
    axs[0].set_ylabel("Gas per alert")
    axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axs[0].set_xlabel("(a)")
    lat = [c["operations"]["anchor_alert"]["latency_ms_mean"], c["operations"]["anchor_batch"]["latency_ms_mean"],
           REP["b15_system"]["block_ms"]]
    b = axs[1].bar([f"{PROP}\nper alert", f"{PROP}\nMerkle batch", f"{N15}\n(reported)"], lat,
                   color=[C_PROP, "#8fb8d9", C_PUB], width=0.55)
    b[2].set_hatch("///")
    for bb, vv in zip(b, lat):
        axs[1].text(bb.get_x() + bb.get_width() / 2, vv * 1.02, f"{vv:.2f}", ha="center", va="bottom", fontsize=7.5)
    axs[1].set_ylabel("Provenance latency (ms)")
    axs[1].set_xlabel("(b)")
    for ax in axs:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig12_blockchain")

# ---------------------------------------------------------------- Fig 13: device datasets
if not iot.empty:
    tasks = [("ton_iot", "holdout", "ToN-IoT"), ("ciciomt2024", "binary", "CICIoMT2024\nbinary"),
             ("ciciomt2024", "device_insider", "CICIoMT2024\ncompromised\ndevice"),
             ("ciciomt2024", "category", "CICIoMT2024\n6-class")]
    series = [("cloud_lgbm", f"{PROP} (cloud tier)", C_PROP, None), ("fog_lgbm", f"{PROP} (fog tier)", "#8fb8d9", None),
              ("xgb_base15", f"{N15} (re-run)", C_ENS, None)]
    x, w = np.arange(len(tasks)), 0.2
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, (mdl, lab, col, _) in enumerate(series):
        mu, sd = [], []
        for ds, pr, _ in tasks:
            q = iot[(iot.dataset == ds) & (iot.protocol == pr) & (iot.model == mdl)].f1_macro
            mu.append(q.mean() if len(q) else np.nan)
            sd.append(q.std(ddof=1) if len(q) > 1 else 0)
        b = ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=2, color=col, label=lab, error_kw=dict(lw=0.8))
        for bb, vv, ee in zip(b, mu, sd):
            if np.isfinite(vv):
                ax.text(bb.get_x() + bb.get_width() / 2, vv + ee + 0.015, f"{vv:.3f}", ha="center", fontsize=6,
                        rotation=90, va="bottom", color="#333333")
    pb = ax.bar([x[0] + 1.5 * w], [REP["b15_toniot_cv"]["f1_macro"]], w, color=C_PUB,
                label=f"{N15} (reported, 5-fold CV)", **PUB)
    ax.text(pb[0].get_x() + w / 2, REP["b15_toniot_cv"]["f1_macro"] + 0.015, "1.000", ha="center", fontsize=6,
            rotation=90, va="bottom", color="#333333")
    ax.set_xticks(x, [t[2] for t in tasks], fontsize=8)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 1.18)       # bars start at zero (no truncated axis)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.5)
    save(fig, "fig13_device_datasets")
print("done")

# PATCH_V11
