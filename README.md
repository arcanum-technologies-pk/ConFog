# ConFog: Context-Aware Fog–Cloud Insider Threat Detection with Tamper-Evident Alert Records

Code, configurations and results for the paper *ConFog: Context-Aware Fog–Cloud Insider Threat Detection
with Tamper-Evident Alert Records for Smart Healthcare* (MDPI *Computers*, submitted).

A gateway in each hospital ward scores every user-day with a fog model (LightGBM on behaviour, deviation and
context features plus isolation-forest and autoencoder scores), closes normal days locally and forwards
suspicious days to the cloud (14-day GRU sequence model and shared-computer graph). Alert thresholds depend on
role and off-hours activity. A keyed fingerprint (HMAC-SHA-256) of each alert is anchored on a permissioned ledger.

## Repository layout
| Path | Content |
|---|---|
| `src/cert_features.py` | user-day features, causal deviations, context features, role mapping |
| `src/splits.py` | chronological, user-disjoint, leave-one-scenario-out and cross-release protocols |
| `src/models.py`, `src/graph.py` | isolation forest, autoencoder, LightGBM fog model, GRU + graph cloud model |
| `src/pipeline.py` | ConFog (fog–cloud fusion, context-adaptive thresholds) and the re-run baselines (RF, HybridXAI) |
| `src/experiments.py` | all CERT and device-traffic (ToN-IoT, CICIoMT2024) experiments |
| `src/fog_node.py`, `src/fog_runtime.py`, `src/bench.py` | fog gateway with adaptive forwarding; Docker testbed |
| `src/chain.py`, `blockchain/`, `scripts/besu_net.py` | AccessPolicy, DeviceRegistry, AlertAnchor; Hardhat and 4-validator Hyperledger Besu ledgers |
| `src/review_extra.py` | model size/latency trade-off, mimicry insiders, faithfulness of the alert reasons |
| `src/paper_outputs.py`, `scripts/paper_figures.py` | every table and result figure of the paper |
| `configs/` | testbed and ledger configurations (each run writes to its own `results/` folder) |
| `results/` | all results used in the paper (see below) |

## Requirements
- Ubuntu 22.04/24.04, Python 3.12; NVIDIA GPU optional (falls back to CPU)
- `pip install -r requirements.txt` (exact versions used: `requirements-lock.txt`)
- Docker (testbed and Besu), `tc-netem` (iproute2), Node.js 18+ (`cd blockchain && npm install`)
- Energy: read access to `/sys/class/powercap/intel-rapl*/energy_uj` (Intel CPU; otherwise energy is skipped)
- Optional: TeX Gyre Pagella font (`fonts-texgyre`) so that the figures use the paper's font
- Reference hardware: Intel i7-14700, 64 GB RAM, NVIDIA RTX 4000 Ada (20 GB)

## Data (not redistributed; licences of the providers apply)
| Dataset | Source | Place in |
|---|---|---|
| CERT insider threat r4.2, r5.2, r6.2 | Software Engineering Institute, Carnegie Mellon University (KiltHub) | `data/raw/cert/<release>/` |
| ToN-IoT (network) | UNSW Canberra | `data/raw/ton_iot/` |
| CICIoMT2024 | Canadian Institute for Cybersecurity | `data/raw/ciciomt2024/` |

## Check the paper's numbers without the data
`results/` contains every run used in the paper. This rebuilds all tables and result figures from it:
```bash
python3 reproduce.py paper     # -> results/paper/PAPER_TABLES.md, one CSV per table, results/paper_figures/
```

## Reproduce everything
```bash
python3 reproduce.py all                 # every stage below, in order
python3 reproduce.py cert bench          # any subset
```
Every stage resumes (finished runs are skipped). Seeds 5–9. No hyperparameter or threshold is chosen on test data.

| Stage | What it runs | Paper |
|---|---|---|
| `features` | user-day feature tables for r4.2, r5.2, r6.2 | Section 4, Table 7 |
| `cert` | all protocols, baselines, ablations, leakage checks, deep baseline | Tables 9–13, Figures 5–9 |
| `iot` | ToN-IoT and CICIoMT2024 device-traffic models | Appendix A |
| `explain` | Shapley values and alert reasons of the fog model | Figure 10 |
| `bench` | testbed: full test set, energy (RAPL), realistic load, saturated gateways | Tables 14–15, Section 6.8 |
| `chain` | ledger on Hardhat and on a 4-validator Besu network (QBFT, permissioned) | Table 16, Section 6.9 |
| `review` | model size vs latency, mimicry insiders, faithfulness test | Sections 6.4, 6.7, 6.8 |
| `report`, `paper` | summary report; all tables and result figures | all |

The keyed alert fingerprint needs a secret: `export HOSPITAL_KEY=<any random string>` before the `chain` stage.

## Licence
Code: MIT (see `LICENSE`). Results: CC BY 4.0. The datasets keep their own licences and are not included.

## Citation
If you use this code, please cite the paper (citation will be added after publication).
