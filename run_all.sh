#!/usr/bin/env bash
# Runs the three experiments, then rebuilds the tables and figures.
# Usage: bash run_all.sh [config]      (default: configs/default.yaml; results go to the config's output_dir)
set -euo pipefail
cd "$(dirname "$0")"
CFG=${1:-configs/default.yaml}
OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")

python -W ignore -m src.run_retrieval --config "$CFG"           # Experiment 1: TREC-COVID
python -W ignore -m src.run_qa --config "$CFG" --stage ablation  # Experiment 3: development articles
python -W ignore -m src.run_qa --config "$CFG" --stage main      # Experiment 2: test articles
python -m src.make_figures --root "$OUT"                         # tables and summary figures
pip freeze > "$OUT/environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt" 2>/dev/null || true
