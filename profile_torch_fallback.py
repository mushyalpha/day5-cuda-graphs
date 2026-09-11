from __future__ import annotations

import argparse
import json
import re
import sys
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baby-vllm"))
from static_engine import (
    StaticDecodeEngine, BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket, ctx_bucket_for,
)

CATEGORY_PATTERNS = [
    ("attention", re.compile(r"attn|attention|sdpa|scaled_dot_product|fmha|flash", re.I)),
    ("linear",    re.compile(r"linear|matmul|\bmm\b|addmm|bmm|gemm", re.I)),
    ("norm_rope", re.compile(r"norm|rope|rotary|rms", re.I)),
]

def categorise(name: str) -> str:
    for cat, pat in CATEGORY_PATTERNS:
        if pat.search(name):
            return cat
    return "other"

def _cuda_time(event) -> float:
    for attr in ("self_cuda_time_total", "self_device_time_total"):
        v = getattr(event, attr, None)
        if v is not None:
            return float(v)
    return 0.0

@torch.inference_mode()
def profile_path(engine, batch_size, ctx_len, mode, warmup=10, profiled=20):
    from bench_graph import prefill_bvllm
    bb = ceil_to_bucket(batch_size, BATCH_BUCKETS)
    cb = ctx_bucket_for(ctx_len)
    buf = engine._ensure_static_buffers(bb, cb)
    state = prefill_bvllm(engine, batch_size, ctx_len)

    def step():
        engine.prepare_static_inputs(
            token_ids=state["next_tokens"],
            positions=state["positions"],
            block_tables=state["block_tables"],
            seq_lens=state["seq_lens"],
            slot_mappings=state["slot_mappings"],
            batch_bucket=bb, ctx_bucket=cb,
        )
        if mode == "graph":
            engine.replay(buf, bb, cb)
        else:
            engine._forward_static(buf, bb, cb)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(profiled):
            step()
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

    events = prof.key_averages()
    total_cuda_us = sum(_cuda_time(e) for e in events)
    busy_pct = min(100.0, 100.0 * (total_cuda_us / 1e6) / wall_s) if wall_s > 0 else 0.0

    cat_totals: dict[str, float] = {}
    for e in events:
        cat = categorise(e.key)
        cat_totals[cat] = cat_totals.get(cat, 0.0) + _cuda_time(e)
    grand = sum(cat_totals.values()) or 1.0
    breakdown = {cat: round(100.0 * v / grand, 1) for cat, v in cat_totals.items()}

    return round(busy_pct, 1), breakdown

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output", default="profile_day5.json")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ctx",   type=int, default=128)
    args = p.parse_args()

    engine = StaticDecodeEngine.from_pretrained(args.model)
    engine.capture_all()

    result = {}
    for mode in ["eager", "graph"]:
        actual_mode = "static_eager" if mode == "graph" else mode
        print(f"Profiling {mode} (actual: {actual_mode}) …")
        busy, breakdown = profile_path(engine, args.batch, args.ctx, mode)
        result[mode] = {
            "gpu_busy_fraction": busy,
            "time_breakdown": {str(args.ctx): breakdown},
            "kernel_launches_per_step": -1,
            "kernels_per_step": -1,
            "note": (
                "torch.profiler fallback — graph path profiled as static_eager proxy. "
                "For accurate per-kernel breakdown inside graph replays, use nsys "
                "with --cuda-graph-trace=node."
            ),
        }
        print(f"  {mode}: busy={busy}%, breakdown={breakdown}")

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
