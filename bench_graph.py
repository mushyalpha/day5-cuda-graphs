from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baby-vllm"))

from babyvllm.config import ModelConfig, CacheConfig, SchedulerConfig
from babyvllm.worker.model_runner import ModelRunner
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.worker.loader import load_model as bvllm_load
from babyvllm.layers.attention import Attention

from static_engine import (
    StaticDecodeEngine, StaticBuffers, BATCH_BUCKETS, CTX_BUCKETS,
    ceil_to_bucket, ctx_bucket_for,
)
from static_attention import StaticDecodeAttention

COMPILE_WARMUP = 20

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "day4-roofline"))
from energy import EnergyTracker, cost_per_million_tokens

def decode_floor_ms(
    n_params: int, n_layers: int, n_kv_heads: int, head_dim: int,
    batch: int, ctx: int, bw_tb_s: float = 3.35,
    bytes_per_param: int = 2,
) -> float:
    weight_bytes = n_params * bytes_per_param
    kv_bytes = batch * ctx * 2 * n_layers * n_kv_heads * head_dim * bytes_per_param
    total = weight_bytes + kv_bytes
    return (total / (bw_tb_s * 1e12)) * 1e3

@torch.inference_mode()
def prefill_bvllm(engine: StaticDecodeEngine, batch_size: int, ctx_len: int) -> dict:

    from babyvllm.sequence import Sequence
    from babyvllm.kv_cache_manager import KVCacheManager

    kv_mgr = KVCacheManager(engine.num_blocks, engine.block_size)
    sequences = []

    for _ in range(batch_size):
        tokens = torch.randint(100, 50000, (ctx_len,)).tolist()
        seq = Sequence(token_ids=tokens)
        kv_mgr.allocate_slots(seq, ctx_len)
        sequences.append(seq)

    all_slots = []
    all_tokens = []
    all_positions = []
    query_start_loc_cpu = [0]
    seq_lens_cpu = []
    context_lens_cpu = []
    max_blocks = max(len(kv_mgr.get_block_table(s)) for s in sequences)
    padded_bt = []

    for seq in sequences:
        qlen = len(seq.token_ids)
        all_tokens.extend(seq.token_ids)
        all_positions.extend(range(qlen))
        all_slots.extend(kv_mgr.get_slot_mapping(seq, qlen))
        query_start_loc_cpu.append(query_start_loc_cpu[-1] + qlen)
        seq_lens_cpu.append(qlen)
        context_lens_cpu.append(0)
        bt = kv_mgr.get_block_table(seq)
        padded_bt.append(bt + [0] * (max_blocks - len(bt)))

    dev = engine.device
    attn_meta = AttentionMetadata(
        slot_mapping    = torch.tensor(all_slots,            dtype=torch.int64,  device=dev),
        block_tables    = torch.tensor(padded_bt,            dtype=torch.int32,  device=dev),
        query_start_loc = torch.tensor(query_start_loc_cpu, dtype=torch.int32,  device=dev),
        seq_lens        = torch.tensor(seq_lens_cpu,        dtype=torch.int32,  device=dev),
        context_lens    = torch.tensor(context_lens_cpu,    dtype=torch.int32,  device=dev),
        max_query_len   = ctx_len,
        max_seq_len     = ctx_len,
        query_start_loc_cpu = query_start_loc_cpu,
        seq_lens_cpu        = seq_lens_cpu,
        context_lens_cpu    = context_lens_cpu,
    )

    input_ids = torch.tensor(all_tokens,     dtype=torch.int64, device=dev)
    positions  = torch.tensor(all_positions, dtype=torch.int64, device=dev)

    for m in engine.model.modules():
        if isinstance(m, StaticDecodeAttention):
            m.__class__ = Attention

    with set_forward_context(attn_meta):
        hidden = engine.model(input_ids, positions)

    for m in engine.model.modules():
        if isinstance(m, Attention) and not isinstance(m, StaticDecodeAttention):
            m.__class__ = StaticDecodeAttention
            m._ctx_bucket = max(CTX_BUCKETS)
            m._block_size  = engine.block_size

    last_indices = torch.tensor(
        [query_start_loc_cpu[i + 1] - 1 for i in range(batch_size)], device=dev,
    )
    next_tokens = engine.model.compute_logits(hidden[last_indices]).argmax(dim=-1).cpu().tolist()

    for seq in sequences:
        seq.advance_computed(len(seq.token_ids))
        kv_mgr.allocate_slots(seq, 1)

    new_slots     = [kv_mgr.get_slot_mapping(seq, 1)[0] for seq in sequences]
    new_seq_lens  = [ctx_len + 1] * batch_size
    new_positions = [ctx_len] * batch_size
    block_tables  = [kv_mgr.get_block_table(seq) for seq in sequences]

    return {
        "next_tokens":   next_tokens,
        "positions":     new_positions,
        "seq_lens":      new_seq_lens,
        "block_tables":  block_tables,
        "slot_mappings": new_slots,
        "sequences":     sequences,
        "kv_mgr":        kv_mgr,
    }

def advance_state(state: dict, next_toks: list[int]) -> dict:
    kv_mgr = state["kv_mgr"]
    sequences_inner = state["sequences"]
    new_slots = []
    for i, seq in enumerate(sequences_inner):
        seq.append_token(next_toks[i])
        seq.advance_computed(1)
        kv_mgr.allocate_slots(seq, 1)
        new_slots.append(kv_mgr.get_slot_mapping(seq, 1)[0])
    return {
        "next_tokens": next_toks,
        "positions":   [p + 1 for p in state["positions"]],
        "seq_lens":    [sl + 1 for sl in state["seq_lens"]],
        "block_tables": [kv_mgr.get_block_table(seq) for seq in sequences_inner],
        "slot_mappings": new_slots,
        "sequences":    sequences_inner,
        "kv_mgr":       kv_mgr,
    }

def time_decode_steps(fn, warmup: int = 10, timed: int = 50) -> list[float]:

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    return timings

@torch.inference_mode()
def verify_correctness(engine: StaticDecodeEngine, model_path: str, n_steps: int = 64):

    print("\n=== Correctness Verification ===")

    if not engine._graphs:
        print("Capturing graphs for verification …")
        engine.capture_all()

    test_cases = [
        (1, 128), (2, 100), (4, 200), (8, 512),
    ]
    all_passed = True

    for batch_size, ctx_len in test_cases:
        bb = ceil_to_bucket(batch_size, BATCH_BUCKETS)
        cb = ctx_bucket_for(ctx_len, n_steps)
        buf = engine._ensure_static_buffers(bb, cb)

        state = prefill_bvllm(engine, batch_size, ctx_len)

        eager_tokens = []
        graph_tokens = []

        for step in range(n_steps):
            engine.prepare_static_inputs(
                token_ids=state["next_tokens"],
                positions=state["positions"],
                block_tables=state["block_tables"],
                seq_lens=state["seq_lens"],
                slot_mappings=state["slot_mappings"],
                batch_bucket=bb,
                ctx_bucket=cb,
            )

            engine._forward_static(buf, bb, cb)
            eager_toks = buf.next_tokens[:batch_size].cpu().tolist()

            engine.replay(buf, bb, cb)
            graph_toks = buf.next_tokens[:batch_size].cpu().tolist()

            if eager_toks != graph_toks:
                print(f"  FAIL B={batch_size} ctx={ctx_len} step={step}: "
                      f"eager={eager_toks} graph={graph_toks}")
                all_passed = False

            eager_tokens.append(eager_toks)
            graph_tokens.append(graph_toks)

            state = advance_state(state, eager_toks)

        print(f"  B={batch_size} ctx={ctx_len}: {'PASS' if eager_tokens == graph_tokens else 'FAIL'} "
              f"({n_steps} steps)")

    print("  Adversarial bucket-switch: B=4/ctx=256 immediately after B=1/ctx=256 …")
    state1 = prefill_bvllm(engine, 1, 128)
    engine.prepare_static_inputs(
        token_ids=state1["next_tokens"], positions=state1["positions"],
        block_tables=state1["block_tables"], seq_lens=state1["seq_lens"],
        slot_mappings=state1["slot_mappings"], batch_bucket=1, ctx_bucket=256,
    )
    engine.replay(engine._ensure_static_buffers(1, 256), 1, 256)
    tok1_after = engine._ensure_static_buffers(1, 256).next_tokens[:1].cpu().tolist()

    state4 = prefill_bvllm(engine, 4, 128)
    engine.prepare_static_inputs(
        token_ids=state4["next_tokens"], positions=state4["positions"],
        block_tables=state4["block_tables"], seq_lens=state4["seq_lens"],
        slot_mappings=state4["slot_mappings"], batch_bucket=4, ctx_bucket=256,
    )
    engine.replay(engine._ensure_static_buffers(4, 256), 4, 256)
    tok4_after = engine._ensure_static_buffers(4, 256).next_tokens[:4].cpu().tolist()
    print(f"    B=1 output: {tok1_after}, B=4 output: {tok4_after} "
          f"(check: should be independent)")

    if all_passed:
        print("  ALL CORRECTNESS TESTS PASSED ✓")
    else:
        print("  CORRECTNESS FAILURES DETECTED ✗")
        sys.exit(1)
    return all_passed

@torch.inference_mode()
def measure_padding_waste(engine: StaticDecodeEngine) -> dict:

    results = {}
    for actual_b, bucket_b in [(3, 4), (33, 64)]:
        ctx = 128
        cb = ctx_bucket_for(ctx, 35)
        buf_full = engine._ensure_static_buffers(bucket_b, cb)
        buf_actual = engine._ensure_static_buffers(actual_b, cb)

        engine.capture_one(actual_b, cb)

        state = prefill_bvllm(engine, actual_b, ctx)
        for bucket in (bucket_b, actual_b):
            engine.prepare_static_inputs(
                token_ids=state["next_tokens"],
                positions=state["positions"],
                block_tables=state["block_tables"],
                seq_lens=state["seq_lens"],
                slot_mappings=state["slot_mappings"],
                batch_bucket=bucket,
                ctx_bucket=cb,
            )

        t_full = time_decode_steps(
            lambda: engine.replay(buf_full, bucket_b, cb),
            warmup=5, timed=30,
        )
        t_actual = time_decode_steps(
            lambda: engine.replay(buf_actual, actual_b, cb),
            warmup=5, timed=30,
        )

        results[f"B{actual_b}_vs_bucket{bucket_b}"] = {
            "actual_batch": actual_b,
            "bucket_batch": bucket_b,
            "padded_rows": bucket_b - actual_b,
            "padded_pct": round(100 * (bucket_b - actual_b) / bucket_b, 1),
            "actual_median_ms": round(float(np.median(t_actual)), 3),
            "bucketed_median_ms": round(float(np.median(t_full)), 3),
            "overhead_ms": round(float(np.median(t_full) - np.median(t_actual)), 3),
            "overhead_pct": round(
                100 * (np.median(t_full) - np.median(t_actual)) / np.median(t_actual), 1
            ),
        }
        print(f"  Padding waste B={actual_b}→{bucket_b}: "
              f"{results[f'B{actual_b}_vs_bucket{bucket_b}']['overhead_pct']:.1f}% overhead "
              f"({results[f'B{actual_b}_vs_bucket{bucket_b}']['padded_pct']:.0f}% padded rows)")

    return results

@torch.inference_mode()
def run_benchmark(
    engine: StaticDecodeEngine,
    hf_results: Optional[dict],
    spec: dict,
    batch_sizes: list[int],
    ctx_lengths: list[int],
    warmup: int = 10,
    timed: int = 50,
    hbm_bw: float = 3.35,
    run_compile: bool = True,
) -> list[dict]:

    all_results = []

    for bs in batch_sizes:
        for ctx in ctx_lengths:
            print(f"\n▸ B={bs}, ctx={ctx} …", flush=True)
            bb = ceil_to_bucket(bs, BATCH_BUCKETS)
            col_warmup = max(warmup, COMPILE_WARMUP) if run_compile else warmup
            cb = ctx_bucket_for(ctx, col_warmup + timed)

            hf_ms = None
            if hf_results:
                for r in hf_results.get("results", []):
                    if r["batch_size"] == bs and r["context_length"] == ctx:
                        hf_ms = r["median_ms"]
                        break

            buf = engine._ensure_static_buffers(bb, cb)

            def make_step_fn(use_graph: bool):
                _state = prefill_bvllm(engine, bs, ctx)

                def step():
                    nonlocal _state
                    engine.prepare_static_inputs(
                        token_ids=_state["next_tokens"],
                        positions=_state["positions"],
                        block_tables=_state["block_tables"],
                        seq_lens=_state["seq_lens"],
                        slot_mappings=_state["slot_mappings"],
                        batch_bucket=bb,
                        ctx_bucket=cb,
                    )
                    if use_graph:
                        engine.replay(buf, bb, cb)
                    else:
                        engine._forward_static(buf, bb, cb)
                    next_toks = buf.next_tokens[:bs].cpu().tolist()
                    _state = advance_state(_state, next_toks)

                return step

            print("  day3_eager …", end="", flush=True)
            try:
                try:
                    _state_day3 = prefill_bvllm(engine, bs, ctx)
                    for m in engine.model.modules():
                        if isinstance(m, StaticDecodeAttention):
                            m.__class__ = Attention

                    def day3_step():
                        nonlocal _state_day3

                        engine.prepare_static_inputs(
                            token_ids=_state_day3["next_tokens"],
                            positions=_state_day3["positions"],
                            block_tables=_state_day3["block_tables"],
                            seq_lens=_state_day3["seq_lens"],
                            slot_mappings=_state_day3["slot_mappings"],
                            batch_bucket=bb, ctx_bucket=cb,
                        )
                        engine._forward_static(buf, bb, cb)
                        next_toks = buf.next_tokens[:bs].cpu().tolist()
                        _state_day3 = advance_state(_state_day3, next_toks)

                    t_day3 = time_decode_steps(day3_step, warmup=warmup, timed=timed)
                    day3_ms = float(np.median(t_day3))
                    print(f" {day3_ms:.2f} ms", flush=True)
                finally:
                    for m in engine.model.modules():
                        if isinstance(m, Attention) and not isinstance(m, StaticDecodeAttention):
                            m.__class__ = StaticDecodeAttention
                            m._ctx_bucket = max(CTX_BUCKETS)
                            m._block_size  = engine.block_size
            except Exception as ex:
                print(f" ERROR: {ex}")
                day3_ms = None

            print("  static_eager …", end="", flush=True)
            try:
                t_static = time_decode_steps(make_step_fn(use_graph=False), warmup=warmup, timed=timed)
                static_ms = float(np.median(t_static))
                print(f" {static_ms:.2f} ms", flush=True)
            except Exception as ex:
                print(f" ERROR: {ex}")
                static_ms = None

            print("  graph …", end="", flush=True)
            try:
                t_graph = time_decode_steps(make_step_fn(use_graph=True), warmup=warmup, timed=timed)
                graph_ms = float(np.median(t_graph))
                print(f" {graph_ms:.2f} ms", flush=True)
            except Exception as ex:
                print(f" ERROR: {ex}")
                graph_ms = None

            compile_ms = None
            if run_compile:
                print("  compile …", end="", flush=True)
                try:
                    import torch._dynamo
                    torch._dynamo.reset()

                    raw_forward_static = engine._forward_static.__wrapped__
                    compiled_forward = torch.compile(
                        raw_forward_static,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    _state_c = prefill_bvllm(engine, bs, ctx)

                    def compile_step():
                        nonlocal _state_c
                        engine.prepare_static_inputs(
                            token_ids=_state_c["next_tokens"],
                            positions=_state_c["positions"],
                            block_tables=_state_c["block_tables"],
                            seq_lens=_state_c["seq_lens"],
                            slot_mappings=_state_c["slot_mappings"],
                            batch_bucket=bb, ctx_bucket=cb,
                        )
                        torch.compiler.cudagraph_mark_step_begin()
                        compiled_forward(engine, buf, bb, cb)
                        next_toks = buf.next_tokens[:bs].cpu().tolist()
                        _state_c = advance_state(_state_c, next_toks)

                    t_compile = time_decode_steps(compile_step, warmup=max(warmup, COMPILE_WARMUP), timed=timed)
                    compile_ms = float(np.median(t_compile))
                    print(f" {compile_ms:.2f} ms", flush=True)
                except Exception as ex:
                    print(f" ERROR: {ex}")

            floor_ms = decode_floor_ms(
                spec["n_params"], spec["n_layers"], spec["n_kv_heads"], spec["head_dim"],
                bs, ctx, hbm_bw,
            )

            def pct_floor(ms):
                if ms is None:
                    return None
                return round(floor_ms / ms * 100, 1)

            def speedup(baseline_ms, improved_ms):
                if baseline_ms and improved_ms:
                    return round(baseline_ms / improved_ms, 2)
                return None

            row = {
                "batch_size": bs,
                "context_length": ctx,
                "batch_bucket": bb,
                "ctx_bucket": cb,
                "floor_ms": round(floor_ms, 3),

                "hf_eager_ms": round(hf_ms, 3) if hf_ms else None,
                "day3_eager_ms": round(day3_ms, 3) if day3_ms else None,
                "static_eager_ms": round(static_ms, 3) if static_ms else None,
                "graph_ms": round(graph_ms, 3) if graph_ms else None,
                "compile_ms": round(compile_ms, 3) if compile_ms else None,

                "hf_pct_floor": pct_floor(hf_ms),
                "day3_pct_floor": pct_floor(day3_ms),
                "static_pct_floor": pct_floor(static_ms),
                "graph_pct_floor": pct_floor(graph_ms),
                "compile_pct_floor": pct_floor(compile_ms),

                "static_vs_day3": speedup(day3_ms, static_ms),
                "graph_vs_day3": speedup(day3_ms, graph_ms),
                "graph_vs_static": speedup(static_ms, graph_ms),
                "compile_vs_day3": speedup(day3_ms, compile_ms),

                "day3_all_ms": [round(x, 4) for x in t_day3] if day3_ms else [],
                "static_all_ms": [round(x, 4) for x in t_static] if static_ms else [],
                "graph_all_ms": [round(x, 4) for x in t_graph] if graph_ms else [],
            }
            all_results.append(row)

    return all_results

def print_table(results: list[dict]):
    print(f"\n{'='*130}")
    print(f"{'Case':<18} {'Floor':>7} {'HF eager':>10} {'day3':>10} "
          f"{'static':>10} {'graph':>10} {'compile':>10} "
          f"{'graph/static':>13} {'graph % floor':>14}")
    print(f"{'='*130}")
    for r in results:
        label = f"B={r['batch_size']}, ctx={r['context_length']}"
        hf    = f"{r['hf_eager_ms']:.2f}"    if r["hf_eager_ms"]    else "  —  "
        day3  = f"{r['day3_eager_ms']:.2f}"  if r["day3_eager_ms"]  else "  —  "
        static = f"{r['static_eager_ms']:.2f}" if r["static_eager_ms"] else "  —  "
        graph  = f"{r['graph_ms']:.2f}"       if r["graph_ms"]       else "  —  "
        comp   = f"{r['compile_ms']:.2f}"     if r["compile_ms"]     else "  —  "
        gs     = f"{r['graph_vs_static']:.2f}×" if r["graph_vs_static"] else "  —  "
        gpct   = f"{r['graph_pct_floor']:.0f}%" if r["graph_pct_floor"] else "  —  "
        print(f"{label:<18} {r['floor_ms']:>7.2f} {hf:>10} {day3:>10} "
              f"{static:>10} {graph:>10} {comp:>10} {gs:>13} {gpct:>14}")
    print(f"{'='*130}")
    print("All times in ms.  Speedup = static_eager / graph (pure launch overhead win).")

def parse_args():
    p = argparse.ArgumentParser(description="Day 5 decode benchmark: eager vs CUDA graphs")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--hf-results", default=None,
                   help="Path to Day 4 results_decode.json for HF column")
    p.add_argument("--batch-sizes", default="1,4,16,64")
    p.add_argument("--ctx-lengths", default="128,2048")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--timed-steps", type=int, default=50)
    p.add_argument("--hbm-bw", type=float, default=3.35)
    p.add_argument("--output", default="results_day5.json")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile column (useful if torch version < 2.5)")
    p.add_argument("--nsys-mode", choices=["eager", "graph"], default=None,
                   help="Run a single instrumented decode loop (NVTX-wrapped "
                        "per step) for nsys profiling instead of the full "
                        "benchmark matrix. Uses the first --batch-sizes / "
                        "--ctx-lengths value.")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip greedy token-id check (diagnostic remasures)")
    return p.parse_args()

@torch.inference_mode()
def run_nsys_probe(engine: StaticDecodeEngine, mode: str, batch_size: int, ctx_len: int,
                    warmup: int = 5, timed: int = 20):
    bb = ceil_to_bucket(batch_size, BATCH_BUCKETS)
    cb = ctx_bucket_for(ctx_len, warmup + timed)
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

    for _ in range(timed):
        torch.cuda.nvtx.range_push("decode_step")
        step()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if (major, minor) < (2, 5):
        print(f"WARNING: torch {torch.__version__} < 2.5; SDPA math backend may not match.")

    engine = StaticDecodeEngine.from_pretrained(args.model)

    print("\nCapturing CUDA graphs …")
    mem_before = torch.cuda.memory_allocated()
    engine.capture_all()
    mem_after = torch.cuda.memory_allocated()
    graph_mem_gb = (mem_after - mem_before) / 1e9
    print(f"Graph memory overhead: {graph_mem_gb:.2f} GB")

    if args.nsys_mode is not None:
        bs0 = int(args.batch_sizes.split(",")[0])
        ctx0 = int(args.ctx_lengths.split(",")[0])
        print(f"\nnsys probe: mode={args.nsys_mode} B={bs0} ctx={ctx0} "
              f"warmup={args.warmup_steps} timed={args.timed_steps}")
        run_nsys_probe(engine, args.nsys_mode, bs0, ctx0,
                       warmup=args.warmup_steps, timed=args.timed_steps)
        return

    if args.verify_only:
        verify_correctness(engine, args.model)
        return

    n_params = sum(p.numel() for p in engine.model.parameters())
    spec = {
        "n_params": n_params,
        "n_layers": engine.n_layers,
        "n_kv_heads": engine.n_kv_heads,
        "head_dim": engine.head_dim,
    }

    hf_results = None
    hf_path = args.hf_results or os.path.join(
        os.path.dirname(__file__), "..", "day4-roofline", "results_decode.json"
    )
    if os.path.exists(hf_path):
        with open(hf_path) as f:
            hf_results = json.load(f)
        print(f"Loaded HF Day 4 results from {hf_path}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    ctx_lengths  = [int(x) for x in args.ctx_lengths.split(",")]

    if not args.skip_verify:
        verify_correctness(engine, args.model)

    print(f"\nRunning benchmark: B∈{batch_sizes} × ctx∈{ctx_lengths}")
    results = run_benchmark(
        engine, hf_results, spec,
        batch_sizes=batch_sizes,
        ctx_lengths=ctx_lengths,
        warmup=args.warmup_steps,
        timed=args.timed_steps,
        hbm_bw=args.hbm_bw,
        run_compile=not args.no_compile,
    )
    print_table(results)

    print("\n=== Padding Waste ===")
    waste = measure_padding_waste(engine)

    out_data = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "hbm_bw_tb_s": args.hbm_bw,
        "graph_memory_gb": round(graph_mem_gb, 3),
        "spec": spec,
        "results": results,
        "padding_waste": waste,
    }
    Path(args.output).write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
