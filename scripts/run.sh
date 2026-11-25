#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export STEP_ROOT="${STEP_ROOT:-$DATA_ROOT/results/StepCluster}"
export RUN_NAME="${RUN_NAME:-qwen25-v1}"
export MODEL_REVISION="${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
export MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen2.5-7B-Instruct/$MODEL_REVISION}"
phase="${1:-help}"
task="${2:-gsm8k}"
split="${3:-test}"
cell="$STEP_ROOT/$RUN_NAME/$task"
pipe=(python -u scripts/pipeline.py)
official() { "${pipe[@]}" author --script "$1" -- "${@:2}"; }
case "$phase" in
  prepare)
    extra=()
    if [[ -n "${DATA_JSONL:-}" ]]; then extra+=(--data-jsonl "$DATA_JSONL"); fi
    if [[ "$task" == gsm8k ]]; then
      extra+=(--dataset-revision "${GSM_REVISION:-740312add88f781978c0658806c59bc2815b9866}")
    fi
    "${pipe[@]}" prepare --task "$task" "${extra[@]}"
    ;;
  generate|extract)
    args=(--task "$task" --split "$split" --model "$MODEL_PATH")
    if [[ "$phase" == generate ]]; then
      args+=(--limit "${LIMIT:-0}" --max-tokens "${MAX_TOKENS:-2048}" --prompt "${PROMPT:-plain}")
    else
      args+=(--state "${STATE:-author-hidden}")
    fi
    "${pipe[@]}" "$phase" "${args[@]}"
    ;;
  smoke)
    export RUN_NAME="${SMOKE_NAME:-qwen25-smoke}"
    LIMIT=8 bash scripts/run.sh generate "$task" test
    bash scripts/run.sh extract "$task" test
    bash scripts/run.sh tsne "$task" test
    ;;
  tsne)
    # The official four-panel publication flag hardcodes a 32-layer Llama.
    # The generic --individual route supports Qwen's actual layer count.
    official scripts/steering/visualize_act_tsne.py --data "$cell/$split/activations.npz" \
      --output-dir "$cell/tsne-$split" --individual
    ;;
  steps)
    official scripts/predictors/train_stepwise_probes.py --data "$cell/$split/activations.npz" \
      --output "$cell/step-probes-$split" --seed "${SEED:-42}"
    ;;
  distances)
    official scripts/trajectories/analyze_step_distances.py --train-npz "$cell/train/activations.npz" \
      --test-npz "$cell/$split/activations.npz" --layer "${LAYER:-27}" --output-dir "$cell/distances-$split"
    ;;
  predict)
    official scripts/predictors/train_predictors.py --npz-path "$cell/train/activations.npz" \
      --output-dir "$cell/predictors-${SEED:-42}" --mode single --label-type correctness \
      --feature-set "${FEATURE:-hash_last_diffs_pca}" --layer "${LAYER:-27}" \
      --seed "${SEED:-42}" --num-workers 1
    ;;
  heldout)
    python -u scripts/evaluate.py --train "$cell/train/activations.npz" \
      --validation "$cell/validation/activations.npz" --test "$cell/test/activations.npz" \
      --output "$cell/heldout-${FEATURE:-late}-${LAYER:-27}" --layer "${LAYER:-27}" --feature "${FEATURE:-late}"
    ;;
  gated|inject)
    [[ "$task" == gsm8k ]] || { echo "Use GSM8K for the official intervention graders" >&2; exit 2; }
    printf -v layer_label '%02d' "${LAYER:-27}"
    predictor="$cell/predictors-${SEED:-42}/hash_minus_last_correctness_layer$layer_label.npz"
    if [[ "$phase" == gated ]]; then
      official scripts/steering/error_aware/intervene_error_aware.py --predictor "$predictor" \
        --steering "$cell/train/activations.npz" --merged-dir "$cell/test/merged" --model "$MODEL_PATH" \
        --mode PROLONG_LAST_N --alpha "${PROLONG_ALPHA:--0.05}" --n-layers 5 \
        --num-questions "${STEER_LIMIT:-20}" --max-new-tokens "${MAX_TOKENS:-2048}" \
        --predictor-threshold "${GATE_THRESHOLD:-0.5}" --output-dir "$cell/gated-${SEED:-42}"
    else
      official scripts/steering/error_aware/intervene_text_injection.py --predictor "$predictor" \
        --merged-dir "$cell/test/merged" --model "$MODEL_PATH" --mode "${INJECT_MODE:-STEP}" \
        --num-questions "${STEER_LIMIT:-20}" --max-new-tokens "${MAX_TOKENS:-2048}" \
        --predictor-threshold "${GATE_THRESHOLD:-0.5}" --output-dir "$cell/inject-${INJECT_MODE:-STEP}-${SEED:-42}"
    fi
    ;;
  ideal)
    # This author's program tunes on TRAIN, using --test-npz only for reporting.
    # The runtime copy is explicitly patched to choose thresholds on validation.
    official scripts/steering/traj_based/compute_ideal_trajectory_thresholds.py \
      --train-npz "$cell/train/activations.npz" --test-npz "$cell/validation/activations.npz" \
      --layer "${LAYER:-27}" --num-steps "${K:-4}" --pca-dim "${PCA_DIM:-128}" \
      --output-dir "$cell/ideal-K${K:-4}-L${LAYER:-27}"
    ;;
  steer)
    [[ "$task" == gsm8k ]] || { echo "Official steering entry grades numeric GSM8K; use GSM8K first" >&2; exit 2; }
    ideal="$cell/ideal-K${K:-4}-L${LAYER:-27}"
    official scripts/steering/traj_based/apply_trajectory_steering.py \
      --model-name "$MODEL_PATH" --ideal-trajectory-model "$ideal/ideal_trajectory_model_K${K:-4}_layer${LAYER:-27}.npz" \
      --thresholds-json "$ideal/thresholds_K${K:-4}_layer${LAYER:-27}.json" \
      --test-npz "$cell/test/activations.npz" --merged-dir "$cell/test/merged" \
      --num-steps "${K:-4}" --layer "${LAYER:-27}" --steering-rank "${RANK:-32}" \
      --alpha "${ALPHA:-0.05}" --max-new-tokens "${MAX_TOKENS:-2048}" \
      --num-questions "${STEER_LIMIT:-50}" --output-dir "$cell/steer-K${K:-4}-L${LAYER:-27}-a${ALPHA:-0.05}"
    ;;
  *) echo "Usage: bash scripts/run.sh {prepare|generate|extract|smoke|tsne|steps|distances|predict|heldout|ideal|steer|gated|inject} [gsm8k|math500|math-full] [train|validation|test]"; exit 2 ;;
esac
