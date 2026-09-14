#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
run_dir="${1:-results/pair-pilot}"
single_run="${2:-results/pilot-verified}"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
test -f "$single_run/features/manifest.json"
test -f "$single_run/decoder/checkpoint-seed-0.pt"
mkdir -p "$run_dir/source"
cp fly_wordbrain/*.py "$run_dir/source/"
"$PYTHON" - "$run_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
names = ['brain.py', 'extract.py', 'pair_brain.py', 'pair_extract.py', 'pair_probe.py']
(run/'extraction-source-lock.json').write_text(json.dumps({name: hashlib.sha256((Path('fly_wordbrain')/name).read_bytes()).hexdigest() for name in names}, indent=2)+'\n')
PY
"$PYTHON" -m fly_wordbrain.pair_probe --output "$run_dir/probe.json"
"$PYTHON" -m fly_wordbrain.pair_extract --dataset data/pilot/dataset.json --output "$run_dir/features" --workers 4
"$PYTHON" - "$run_dir" "$single_run" <<'PY'
import json, sys, time
from pathlib import Path
from fly_wordbrain.pair_decoder import run
output, single = map(Path, sys.argv[1:])
started = time.perf_counter()
run(output/'features', Path('data/pilot/dataset.json'), output/'decoder', single_features=single/'features')
(output/'decoder-timing.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter()-started})+'\n')
PY
"$PYTHON" scripts/audit_pair_pilot.py --run "$run_dir" --dataset data/pilot/dataset.json --single-run "$single_run"
"$PYTHON" -m fly_wordbrain.pair_generate --dataset data/pilot/dataset.json --checkpoint "$run_dir/decoder" --output "$run_dir/generation.json"
