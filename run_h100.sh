#!/usr/bin/env bash
# run_h100.sh — Full Day 5 pipeline.
#
# Ordering:
#   1. 0.5B benchmark  (fast debug; gets speedup-vs-model-size data point)
#   2. 7B benchmark    (the headline numbers)
#   3. nsys profiles   (or torch.profiler fallback)
#   4. Charts
#
# All stages are idempotent: re-run is safe; existing JSON is preserved
# unless FORCE=1.
#
# Environment variables:
#   MODEL_7B   path/to/Qwen2.5-7B   (default: Qwen/Qwen2.5-7B)
#   MODEL_05B  path/to/Qwen2.5-0.5B (default: Qwen/Qwen2.5-0.5B)
#   HF_RESULTS path to Day 4 results_decode.json (optional but recommended)
#   SKIP_05B=1 skip the 0.5B run
#   SKIP_7B=1  skip the 7B run (use existing results_7B.json)
#   FORCE=1    overwrite existing JSON results

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
export PYTHON
MODEL_7B="${MODEL_7B:-Qwen/Qwen2.5-7B}"
MODEL_05B="${MODEL_05B:-Qwen/Qwen2.5-0.5B}"
HF_RESULTS="${HF_RESULTS:-../day4-roofline/results_decode.json}"
FORCE="${FORCE:-0}"

echo "================================================================"
echo "  Day 5/45 · CUDA Graphs · H100 Pipeline"
echo "  7B model:  $MODEL_7B"
echo "  0.5B model: $MODEL_05B"
echo "  HF results: $HF_RESULTS"
echo "================================================================"
echo ""

# ── 1. 0.5B benchmark ────────────────────────────────────────────────
if [ "${SKIP_05B:-0}" = "1" ]; then
    echo "== 1/4: SKIPPED 0.5B (SKIP_05B=1) =="
elif [ -f results_0.5B.json ] && [ "$FORCE" != "1" ]; then
    echo "== 1/4: SKIPPED 0.5B (results_0.5B.json exists; set FORCE=1 to re-run) =="
else
    echo "== 1/4: 0.5B benchmark =="
    EXTRA_HF=""
    if [ -f "$HF_RESULTS" ]; then
        EXTRA_HF="--hf-results $HF_RESULTS"
    fi
    "$PYTHON" bench_graph.py \
        --model "$MODEL_05B" \
        --batch-sizes 1,4,16,64 \
        --ctx-lengths 128,1920 \
        --warmup-steps 10 \
        --timed-steps 50 \
        --output results_0.5B.json \
        $EXTRA_HF
fi

# ── 2. 7B benchmark ──────────────────────────────────────────────────
if [ "${SKIP_7B:-0}" = "1" ]; then
    echo "== 2/4: SKIPPED 7B (SKIP_7B=1) =="
elif [ -f results_7B.json ] && [ "$FORCE" != "1" ]; then
    echo "== 2/4: SKIPPED 7B (results_7B.json exists; set FORCE=1 to re-run) =="
else
    echo "== 2/4: 7B benchmark =="
    EXTRA_HF=""
    if [ -f "$HF_RESULTS" ]; then
        EXTRA_HF="--hf-results $HF_RESULTS"
    fi
    "$PYTHON" bench_graph.py \
        --model "$MODEL_7B" \
        --batch-sizes 1,4,16,64 \
        --ctx-lengths 128,1920 \
        --warmup-steps 10 \
        --timed-steps 50 \
        --output results_7B.json \
        $EXTRA_HF
fi

if [ ! -f results_7B.json ]; then
    echo "FATAL: results_7B.json missing." >&2
    exit 1
fi

# ── 3. Profiling ──────────────────────────────────────────────────────
echo ""
echo "== 3/4: Profiling (nsys if available, torch.profiler fallback) =="
bash profile_nsys.sh \
    --model "$MODEL_7B" \
    --output profile_day5.json \
    --batch 1 \
    --ctx 128 \
    || echo "WARNING: profiling failed; charts 2 will use placeholder data"

# ── 3b. Optional B=64 GQA diagnostic (do NOT re-run the full matrix) ──
# After the q-fold in static_attention.py:
#   nsys profile of the OLD 187 ms cell (confirm repeat_interleave / copy):
#     bash profile_nsys.sh --model "$MODEL_7B" --batch 64 --ctx 1920 --output profile_b64.json
#   Re-measure that one cell with the q-fold (expect 40–70 ms if the 105 GB expand is gone):
#     $PYTHON bench_graph.py --model "$MODEL_7B" --batch-sizes 64 --ctx-lengths 1920 \
#         --no-compile --skip-verify --output results_gqa_fold_b64.json
# Keep the published matrix at 187 ms; report this as a one-line "fixed cell" callout.
echo ""
echo "== 4/4: Generating charts =="
EXTRA_SMALL=""
if [ -f results_0.5B.json ]; then
    EXTRA_SMALL="--data-small results_0.5B.json"
fi
EXTRA_PROFILE=""
if [ -f profile_day5.json ]; then
    EXTRA_PROFILE="--profile profile_day5.json"
fi

"$PYTHON" plot_day5.py \
    --data results_7B.json \
    $EXTRA_SMALL \
    $EXTRA_PROFILE \
    --output-dir charts

echo ""
echo "================================================================"
echo "  DONE.  Pull these files back to your local machine:"
echo "    results_7B.json"
echo "    results_0.5B.json"
echo "    profile_day5.json"
echo "    charts/hero_chart.png"
echo "    charts/chart_*.png"
echo "================================================================"
