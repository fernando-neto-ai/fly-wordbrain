# Launch a unified arm, taking EVERY knob from its own config: rank, plasticity, the task
# set, the corpora and the loss weights.
#
# Three launchers before this one each hardcoded something the config also declared --
# rank 64 in one, the task set in another, the loss weights in the third. A disagreement
# between launcher and config is invisible until an evaluator rebuilds from the config
# hours later, or worse, never surfaces at all and the arm simply trained a different
# objective than its record says.
set -u
cd ~/fly_wordbrain_connectorch
ARM="$1"
CFG="experiments/configs/$ARM.json"
OUT="results/unified-v1/arms/$ARM"
if pgrep -f "train_connectorch|train_multitask|train_unified" >/dev/null; then echo "WORKER PRESENT, aborting"; exit 4; fi
if [ -e "$OUT" ]; then echo "OUTPUT EXISTS: $OUT"; exit 3; fi
test -s "$CFG" || { echo "MISSING CONFIG $CFG"; exit 6; }

read RANK PLAST LEAK SENT TOX TOXBATCH WL WC WS WT WR <<<"$(./.venv-connectorch/bin/python -c "
import json;c=json.load(open('$CFG'));t=c['training'];u=c['unified'];s=u['output_space']
w=u.get('weights',{})
print(t['readout_rank'], t['plasticity'], t.get('leak','fixed'), s.get('sentiment_classes',0),
      u.get('toxicity_corpus','-'), u.get('toxicity_batch',8),
      w.get('language',1.0), w.get('chess',1.0), w.get('sentiment',1.0),
      w.get('toxicity',1.0), w.get('router',0.1))")"
echo "config: rank=$RANK plasticity=$PLAST leak=$LEAK weights lang=$WL chess=$WC sent=$WS tox=$WT router=$WR corpus=$TOX"

ARGS=(--chess-corpus data/chess-positions-v1 --chess-batch 32 --chess-eval-limit 10000
      --language-weight "$WL" --chess-weight "$WC" --router-weight "$WR")
if [ "$SENT" != "0" ]; then
  test -s data/sentiment-sst2-v1/dataset.json || { echo "MISSING SENTIMENT CORPUS"; exit 5; }
  ARGS+=(--sentiment-corpus data/sentiment-sst2-v1 --sentiment-batch 32
         --sentiment-weight "$WS" --sentiment-pooling mean)
fi
if [ "$TOX" != "-" ]; then
  test -s "$TOX/dataset.json" || { echo "MISSING TOXICITY CORPUS $TOX"; exit 5; }
  ARGS+=(--toxicity-corpus "$TOX" --toxicity-batch "$TOXBATCH" --toxicity-weight "$WT")
fi
mkdir -p "$(dirname "$OUT")"
export PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
nohup ./.venv-connectorch/bin/python -u scripts/train_unified.py \
  "${ARGS[@]}" --output "$OUT" \
  --model data/ngxson-fly-llm-hf/65c677b3d566a2e9793d5f72999cdb441c6c0a9f \
  --data data/ngxson-tinystories-10k/dataset.json \
  --groups data/connectorch-groups-v1/groups.npz \
  --device mps --d-embed 32 --readout-rank "$RANK" --plasticity "$PLAST" --leak "$LEAK" --history-length 8 \
  --epochs 4 --second-epochs 2 --max-updates 16800 \
  --batch-size 8 --chunk-size 32 --seed 42 --skip-final-test \
  > ~/unified-$ARM.log 2>&1 &
echo "PID $! ARM $ARM"
