#!/usr/bin/env bash
# profile_nsys.sh — Run nsys on both decode paths and emit profile_day5.json
#
# nsys --cuda-graph-trace=node is required.  Without it, nsys reports each
# graph replay as a single opaque entity and Σ(kernel_time) for the graph
# path collapses "Other" to ~0 artificially.
#
# Two counts are kept distinct:
#   CPU launches per step  (~600 → 1, the quotable number)
#   Kernels executed / step (~600 → ~600, unchanged — physics doesn't care)
#
# NVTX ranges (bench_graph.py uses torch.cuda.nvtx.range_push) let us compute
# per-step wall time and per-step kernel sums rather than averaging over the
# whole run including warmup and capture.
#
# Usage:
#   bash profile_nsys.sh --model Qwen/Qwen2.5-7B --output profile_day5.json
#   NSYS=/opt/nsight-systems/.../nsys bash profile_nsys.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUTPUT="${OUTPUT:-profile_day5.json}"
BATCH="${BATCH:-1}"
CTX="${CTX:-128}"
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)    MODEL="$2";  shift 2 ;;
        --output)   OUTPUT="$2"; shift 2 ;;
        --batch)    BATCH="$2";  shift 2 ;;
        --ctx)      CTX="$2";    shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

print_nsys_install_help() {
    cat <<'EOF'
Need a complete nsys (CLI + QdstrmImporter). Do NOT apt-get install
nsight-systems and do NOT apt-get -f install — both try to overwrite
the RunPod NVIDIA driver via cross-device links.

Tarball is already under /opt/nsight-systems. Point at it explicitly:

  NSYS=$(find /opt/nsight-systems -type f -name nsys | head -1)
  SKIP_05B=1 SKIP_7B=1 NSYS="$NSYS" bash run_h100.sh
EOF
}

find_importer() {
    local nsys_bin="$1"
    local nsys_dir
    nsys_dir="$(cd "$(dirname "$nsys_bin")" && pwd)"
    local c
    for c in \
        "$nsys_dir/QdstrmImporter" \
        "$nsys_dir/../host-linux-x64/QdstrmImporter" \
        "$nsys_dir/../Host-x86_64/QdstrmImporter"
    do
        if [[ -x "$c" ]]; then
            echo "$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"
            return 0
        fi
    done
    c="$(find /opt/nsight-systems -name QdstrmImporter -type f 2>/dev/null | head -1 || true)"
    if [[ -n "$c" && -x "$c" ]]; then
        echo "$c"
        return 0
    fi
    return 1
}

pick_complete_nsys() {
    local cand importer
    if [[ -n "${NSYS:-}" && "$NSYS" != "nsys" ]]; then
        if [[ -x "$NSYS" ]]; then
            echo "$NSYS"
            return 0
        fi
        if command -v "$NSYS" &>/dev/null; then
            echo "$(command -v "$NSYS")"
            return 0
        fi
    fi
    while IFS= read -r cand; do
        [[ -n "$cand" && -x "$cand" ]] || continue
        if importer="$(find_importer "$cand")"; then
            echo "$cand"
            return 0
        fi
    done < <(find /opt/nsight-systems -type f -path '*/host-linux-x64/nsys' 2>/dev/null
             find /opt/nsight-systems -type f -name nsys 2>/dev/null
             command -v nsys 2>/dev/null || true)
    return 1
}

ensure_nsys_rep() {
    local stem="$1"
    if [[ -f "${stem}.nsys-rep" ]]; then
        return 0
    fi
    if [[ ! -f "${stem}.qdstrm" ]]; then
        echo "ERROR: neither ${stem}.nsys-rep nor ${stem}.qdstrm exists" >&2
        return 1
    fi
    echo "Converting ${stem}.qdstrm → ${stem}.nsys-rep"
    "$IMPORTER" -i "${stem}.qdstrm" -o "${stem}.nsys-rep"
    [[ -f "${stem}.nsys-rep" ]]
}

if ! PICKED="$(pick_complete_nsys)"; then
    echo "WARNING: no complete nsys (CLI + QdstrmImporter). Falling back to torch.profiler."
    print_nsys_install_help
    "$PYTHON" profile_torch_fallback.py \
        --model "$MODEL" --output "$OUTPUT" \
        --batch "$BATCH" --ctx "$CTX"
    exit 0
fi
NSYS="$PICKED"
IMPORTER="$(find_importer "$NSYS")"
echo "python:    $(command -v "$PYTHON")"
echo "nsys:      $NSYS"
echo "importer:  $IMPORTER"

profile_one() {
    local mode="$1"
    local stem="$2"
    echo "=== nsys profile: ${mode} path ==="
    "$NSYS" profile \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --stats=true \
        --output="$stem" \
        --force-overwrite=true \
        "$PYTHON" bench_graph.py \
            --model "$MODEL" \
            --batch-sizes "$BATCH" \
            --ctx-lengths "$CTX" \
            --warmup-steps 5 \
            --timed-steps 20 \
            --no-compile \
            --output /dev/null \
            --nsys-mode "$mode"
    ensure_nsys_rep "$stem"
}

profile_one eager nsys_eager
echo ""
profile_one graph nsys_graph

echo ""
echo "=== Exporting nsys reports to sqlite ==="
"$NSYS" export --type=sqlite --force-overwrite=true \
    --output=nsys_eager.sqlite nsys_eager.nsys-rep
"$NSYS" export --type=sqlite --force-overwrite=true \
    --output=nsys_graph.sqlite nsys_graph.nsys-rep

echo ""
echo "=== Post-processing nsys reports → $OUTPUT ==="
"$PYTHON" parse_nsys_day5.py \
    --eager nsys_eager.sqlite \
    --graph nsys_graph.sqlite \
    --output "$OUTPUT"

echo ""
echo "Done. Profile written to $OUTPUT"
