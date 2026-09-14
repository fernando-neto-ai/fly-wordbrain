#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
run_dir="${1:-results/pilot}"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$run_dir/source"
cp fly_wordbrain/*.py "$run_dir/source/"
"$PYTHON" -m fly_wordbrain.probe --output "$run_dir/probe.json"
"$PYTHON" -m fly_wordbrain.extract --dataset data/pilot/dataset.json --output "$run_dir/features" --workers 4
"$PYTHON" -m fly_wordbrain.decoder --dataset data/pilot/dataset.json --features "$run_dir/features" --output "$run_dir/decoder" --seeds 0,1,2
cp "$run_dir/features/manifest.json" "$run_dir/decoder/feature-manifest.json"
"$PYTHON" -m fly_wordbrain.recall --dataset data/pilot/dataset.json --probes data/pilot/recall_probes.json --output "$run_dir/recall" --workers 4
"$PYTHON" -m fly_wordbrain.generate --dataset data/pilot/dataset.json --checkpoint "$run_dir/decoder" --output "$run_dir/generation.json"
