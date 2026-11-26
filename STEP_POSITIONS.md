# Experiment A extension: step boundary and body representations

The independent entry point is `scripts/step_positions.py`. It replays the original token IDs from the completed Experiment A GSM8K run and uses its existing question splits and inclusion audit. The original activation convention is used, including the final normalization for the last layer. Layer indices are zero based. The default layers for Qwen2.5-7B are 0, 9, 18, and 27.

| Sampling family | Implementation |
| --- | --- |
| Boundary | Activation immediately before the Step j marker, as in A |
| Body quantiles | One token each at the 25%, 50%, and 75% indices of eligible tokens within Step j |
| Body mean | Mean across eligible tokens within Step j |
| Nearby control | The nearest eligible token in the preceding step's body that is at least four token positions before the boundary, excluding the preceding body's first two eligible tokens |

Eligible body tokens are fully contained between the header and the next marker or final answer delimiter. Whitespace-only tokens are omitted. Each included row has all six views. The current step must supply at least nine eligible body tokens and the preceding step must supply a control. Step 1 is excluded because its pre-marker state often belongs to the prompt and has no preceding step. The default classification labels are Step 2 through Step 5. The extraction audit records sampled positions and questions with no eligible rows. All compared views use the same qualifying steps within each split.

The boundary is the end of the prefix before entry into Step j. The body states are measured as Step j unfolds. A lower transfer score can reflect a shift between entry and execution. A separate probe per body view measures whether step identity remains decodable with its own readout. The boundary probe is trained and tuned only on boundary states, then applied unchanged to each body view. The paired boundary/control classifier tests whether hidden states distinguish a marker boundary from a nearby ordinary state. Its balanced candidate set is a diagnostic and does not estimate precision when scanning an entire free-form response.

The surface baseline uses absolute position, relative progress, and token identity as a lexical proxy for punctuation and newlines. Body mean uses position features and a constant token-ID indicator. The position baseline uses total generated length for the relative progress feature, so this is an offline study. Upstream markers remain in the causal history. Evidence of decodability here warrants a separate no-marker transfer test before attributing the signal to semantic stages. Select probe regularization by validation macro-F1. Inspect held-out macro-F1, accuracy, and confusion matrices, plus AP for the binary control diagnostic. A high boundary/control score can still reflect formatting.

## Local synchronization

```powershell
cd E:\Research\LLMCoT\methods\StepCluster
git add scripts/step_positions.py STEP_POSITIONS.md
git commit -m "Compare reasoning step boundary and body activations"
git push
```

## Server

```bash
cd ~/projects/StepCluster
conda activate stepcluster
git pull --ff-only
nvidia-smi
export CUDA_VISIBLE_DEVICES=6  # Select an available GPU
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ROOT=/mnt/data/wxh/wdl/results/StepCluster
SOURCE="$ROOT/qwen25-pilot-chat-v2/gsm8k"
OUT="$ROOT/qwen25-pilot-chat-v2/gsm8k/position-probes-v1"
LOG_DIR=/mnt/data/wxh/wdl/logs/StepCluster
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/position-probes-$(date +%Y%m%d-%H%M%S).log"
nohup setsid python -u scripts/step_positions.py all --source "$SOURCE" --output "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$LOG.pid"
tail -n 30 -f "$LOG"
```

The model path defaults to the snapshot recorded by A. Keep about 1 GB available for pilot intermediates and output. Extraction uses GPU forward passes, and probe fitting uses CPU. Completed and verified extraction splits are reused when repeating the same extraction command with the same source, code and settings. An interrupted split is re-extracted. Run only one process per output directory. To rerun only probe fitting, e.g. with Step 2–4 as the label range:

```bash
python -u scripts/step_positions.py analyze --output "$OUT" --max-step 4
```

The log ends with `POSITION PROBES COMPLETE`. See `probes-step2-5.csv` for the comparisons and the accompanying JSON for confusion matrices and class counts. `train-audit.json`, `validation-audit.json`, and `test-audit.json` record the exact sampled positions. For an extraction-only smoke, use the `extract` phase with `--limit 8 --output "$OUT-smoke"`.
