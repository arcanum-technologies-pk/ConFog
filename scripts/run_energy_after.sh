#!/bin/bash
cd ~/ITD-EdgeCloud && source .venv/bin/activate
while pgrep -f "scripts/run_review_queue.sh" >/dev/null; do sleep 60; done
test -r /sys/class/powercap/intel-rapl:0/energy_uj || { echo "RAPL not readable (reboot?) - run the sudo chmod again"; exit 1; }
python3 -u -c "from src.common import load_config, setup_logging; from src.bench import run_bench; cfg=load_config(override='configs/bench_full_rapl.yaml'); setup_logging(cfg['paths']['logs'], 'bench_full_rapl'); run_bench(cfg)"
echo "ENERGY_DONE $(date)"
