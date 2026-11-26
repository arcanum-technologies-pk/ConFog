"""One entry point that reproduces every result of the ConFog paper.

    python3 reproduce.py all            # every stage below, in order
    python3 reproduce.py paper          # only rebuild tables and figures from saved results
    python3 reproduce.py cert bench     # any subset of stages

Stages (each one resumes: finished runs are skipped):
    features  CERT r4.2/r5.2/r6.2 user-day feature tables       -> data/features/*.parquet
    cert      all CERT experiments (protocols, seeds, ablations) -> results/raw, results/scores
    iot       ToN-IoT and CICIoMT2024 device-traffic models     -> results/raw/iot
    explain   SHAP reasons of the fog model                      -> results/explain
    bench     fog-cloud testbed: 10,000-day sample, full test set, saturated gateways (needs Docker)
    chain     permissioned ledger (Hardhat) with keyed fingerprints -> results/chain
    report    summary report of all runs                         -> results/REPORT.md
    paper     every table and result figure of the paper         -> results/paper/
"""
from __future__ import annotations

import sys

from src.common import load_config, setup_logging

BENCH_CONFIGS = ["configs/bench_full.yaml", "configs/bench_full_rapl.yaml", "configs/bench_realistic.yaml", "configs/bench_stress_default.yaml",
                 "configs/bench_stress_aggressive.yaml", "configs/bench_stress_lenient.yaml"]
STAGES = ["features", "cert", "iot", "explain", "bench", "chain", "review", "report", "paper"]


def main(stages: list[str]) -> None:
    cfg = load_config()
    setup_logging(cfg["paths"]["logs"], "reproduce")
    for st in stages:
        if st == "features":
            from src.cert_features import build_features
            for rel in cfg["cert"]["releases"]:
                build_features(cfg, rel)
        elif st == "cert":
            from src.experiments import run_cert
            run_cert(cfg)
        elif st == "iot":
            from src.experiments import run_iot
            run_iot(cfg)
        elif st == "explain":
            from src.explain import run_explain
            run_explain(cfg)
        elif st == "bench":
            from src.bench import run_bench
            run_bench(cfg)
            for c in BENCH_CONFIGS:
                run_bench(load_config(override=c))
        elif st == "chain":
            from src.chain import run_chain
            run_chain(cfg)
            # four-validator Hyperledger Besu network (needs Docker)
            import shutil, subprocess
            from pathlib import Path
            subprocess.run([sys.executable, "scripts/besu_net.py", "up"], check=True)
            try:
                d = Path("results/chain_besu/explain"); d.mkdir(parents=True, exist_ok=True)
                shutil.copy("results/explain/alerts.jsonl", d / "alerts.jsonl")
                run_chain(load_config(override="configs/chain_besu.yaml"))
            finally:
                subprocess.run([sys.executable, "scripts/besu_net.py", "down"])
        elif st == "review":   # reviewer experiments: size/latency trade-off, mimicry insiders, faithfulness
            from src.review_extra import run_faithful, run_mimicry, run_tradeoff
            run_tradeoff(cfg); run_mimicry(cfg); run_faithful(cfg)
        elif st == "report":
            from src.report import run_report
            run_report(cfg)
        elif st == "paper":
            from src.paper_outputs import run_paper_outputs
            run_paper_outputs(cfg)
            # curves, confusion matrices and SHAP figures of the paper
            import subprocess
            from pathlib import Path
            if Path("scripts/paper_figures.py").exists():
                subprocess.run([sys.executable, "scripts/paper_figures.py"], check=True)
        else:
            sys.exit(f"unknown stage '{st}'; choose from: all {' '.join(STAGES)}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["paper"]
    main(STAGES if args == ["all"] else args)
