from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baby-vllm"))

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from babyvllm.config import ModelConfig, CacheConfig, SchedulerConfig
from babyvllm.worker.model_runner import ModelRunner
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.layers.attention import Attention

from static_attention import StaticDecodeAttention

BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
CTX_BUCKETS   = [256, 2048]

DECODE_HEADROOM_BLOCKS = 16
KV_SAFETY_FACTOR = 2

def ceil_to_bucket(value: int, buckets: list[int]) -> int:
    for b in buckets:
        if value <= b:
            return b
    return buckets[-1]

def ctx_bucket_for(ctx_len: int, n_steps: int = 0) -> int:

    needed = ctx_len + 1 + n_steps
    if needed > max(CTX_BUCKETS):
        raise ValueError(
            f"ctx_len {ctx_len} plus {n_steps} decode steps needs {needed} "
            f"positions, past the largest CTX_BUCKET ({max(CTX_BUCKETS)})."
        )
    return ceil_to_bucket(needed, CTX_BUCKETS)

def compute_pool_reserve(
    max_batch_bucket: int,
    max_ctx_bucket: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    bytes_per_element: int = 2,
) -> int:

    gather_per_layer = (
        max_batch_bucket * n_kv_heads * max_ctx_bucket * head_dim
        * 2
        * bytes_per_element
    )
    total_gather = gather_per_layer * n_layers
    slack = 2 * 1024 * 1024 * 1024
    return total_gather + slack

@dataclass
class StaticBuffers:

    input_ids:    torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    seq_lens:     torch.Tensor
    slot_mapping: torch.Tensor
    logits:       torch.Tensor
    next_tokens:  torch.Tensor

class StaticDecodeEngine:

    BLOCK_SIZE = 16

    def __init__(self, model_runner: ModelRunner, model_path: str):
        self.runner = model_runner
        self.model  = model_runner.model
        self.device = model_runner.device
        self.dtype  = model_runner.dtype

        cfg = model_runner.model_config
        self.n_layers    = cfg.num_hidden_layers
        self.n_kv_heads  = cfg.num_key_value_heads
        self.head_dim    = cfg.hidden_size // cfg.num_attention_heads
        self.vocab_size  = cfg.vocab_size
        self.block_size  = self.BLOCK_SIZE

        self._static: dict[tuple[int, int], StaticBuffers] = {}
        self._graphs: dict[tuple[int, int], "CUDAGraphRunner"] = {}
        self._pool = None
        self._kv_allocated = False
        self._seq_lens_cpu: list[int] = []

        max_ctx = max(CTX_BUCKETS)
        self.max_blocks_per_seq = (max_ctx + self.block_size - 1) // self.block_size + 1

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> "StaticDecodeEngine":

        import json, glob
        from babyvllm.worker.loader import load_model as bvllm_load

        if os.path.isdir(model_path):
            local_path = model_path
        else:
            from huggingface_hub import snapshot_download
            print(f"'{model_path}' is not a local directory — resolving via "
                  f"the HF Hub …")
            local_path = snapshot_download(repo_id=model_path)
            print(f"  Checkpoint ready at {local_path}")

        config_path = os.path.join(local_path, "config.json")
        with open(config_path) as f:
            hf_cfg = json.load(f)

        model_config = ModelConfig(
            vocab_size=hf_cfg["vocab_size"],
            hidden_size=hf_cfg["hidden_size"],
            intermediate_size=hf_cfg["intermediate_size"],
            num_hidden_layers=hf_cfg["num_hidden_layers"],
            num_attention_heads=hf_cfg["num_attention_heads"],
            num_key_value_heads=hf_cfg.get("num_key_value_heads", hf_cfg["num_attention_heads"]),
            rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-6),
            max_position_embeddings=hf_cfg.get("max_position_embeddings", 32768),
            block_size=cls.BLOCK_SIZE,
            rope_theta=float(hf_cfg.get("rope_theta", 1_000_000.0)),
            tie_word_embeddings=hf_cfg.get("tie_word_embeddings", False),
        )
        cache_config = CacheConfig(block_size=cls.BLOCK_SIZE)
        sched_config = SchedulerConfig(max_num_batched_tokens=2048)

        runner = ModelRunner(model_config, cache_config, sched_config, device=device)

        print(f"Loading weights from {local_path} …")
        bvllm_load(runner.model, local_path)
        print(f"  Weights loaded.")

        engine = cls(runner, local_path)

        engine._allocate_kv_cache()

        engine._patch_attention()

        return engine

    def _allocate_kv_cache(self):

        if self._kv_allocated:
            return

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()

        pool_reserve = compute_pool_reserve(
            max_batch_bucket=max(BATCH_BUCKETS),
            max_ctx_bucket=max(CTX_BUCKETS),
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            n_layers=self.n_layers,
        )
        available_for_kv = free_bytes - pool_reserve
        bytes_per_block_per_layer = (
            2 * self.block_size * self.n_kv_heads * self.head_dim
            * (2 if self.dtype in (torch.bfloat16, torch.float16) else 4)
        )
        bytes_per_block = bytes_per_block_per_layer * self.n_layers
        n_blocks = max(1, int(available_for_kv // bytes_per_block))

        blocks_per_seq = self.max_blocks_per_seq + DECODE_HEADROOM_BLOCKS
        needed_blocks = max(BATCH_BUCKETS) * blocks_per_seq * KV_SAFETY_FACTOR
        n_blocks_capped = min(n_blocks, needed_blocks)

        print(f"  Free VRAM:        {free_bytes / 1e9:.2f} GB")
        print(f"  Pool reserve:     {pool_reserve / 1e9:.2f} GB")
        print(f"  Available for KV: {available_for_kv / 1e9:.2f} GB")
        if n_blocks_capped < n_blocks:
            print(f"  KV cap:           {n_blocks_capped} blocks "
                  f"({n_blocks_capped * bytes_per_block / 1e9:.2f} GB) — capped from "
                  f"{n_blocks} to leave headroom for compile/profiler scratch")
        n_blocks = n_blocks_capped
        print(f"  KV blocks:        {n_blocks} × {self.block_size} = {n_blocks * self.block_size:,} tokens")

        min_blocks = max(BATCH_BUCKETS) * self.max_blocks_per_seq
        if n_blocks < min_blocks:
            weights_gb = (total_bytes - free_bytes) / 1e9
            raise RuntimeError(
                f"KV budget too small: {n_blocks} blocks available, "
                f"{min_blocks} needed for the largest bucket "
                f"(B={max(BATCH_BUCKETS)} × ctx={max(CTX_BUCKETS)}).\n"
                f"  GPU total {total_bytes / 1e9:.2f} GB, weights and "
                f"context already hold {weights_gb:.2f} GB, leaving "
                f"{free_bytes / 1e9:.2f} GB free.\n"
                f"  Pool reserve for graph capture needs "
                f"{pool_reserve / 1e9:.2f} GB of that, so only "
                f"{available_for_kv / 1e9:.2f} GB is left for the KV cache.\n"
                f"  This model does not fit on this GPU at these bucket "
                f"sizes. Use a larger GPU, or shrink BATCH_BUCKETS / "
                f"CTX_BUCKETS in static_engine.py."
            )

        self.runner.num_blocks = n_blocks
        self.runner._alloc_kv_tensors(n_blocks)
        self.num_blocks = n_blocks
        self._kv_allocated = True

    def _patch_attention(self):

        for m in self.model.modules():
            if isinstance(m, Attention):
                m.__class__ = StaticDecodeAttention
                m._ctx_bucket = max(CTX_BUCKETS)
                m._block_size  = self.block_size

    def _ensure_static_buffers(self, batch_bucket: int, ctx_bucket: int):
        key = (batch_bucket, ctx_bucket)
        if key in self._static:
            return self._static[key]

        B = batch_bucket
        MB = self.max_blocks_per_seq
        V = self.vocab_size
        dev = self.device

        buf = StaticBuffers(
            input_ids    = torch.zeros(B, dtype=torch.int64,  device=dev),
            position_ids = torch.zeros(B, dtype=torch.int64,  device=dev),
            block_tables = torch.zeros(B, MB, dtype=torch.int32, device=dev),
            seq_lens     = torch.zeros(B, dtype=torch.int32,  device=dev),
            slot_mapping = torch.zeros(B, dtype=torch.int64,  device=dev),
            logits       = torch.zeros(B, V, dtype=torch.float32, device=dev),
            next_tokens  = torch.zeros(B, dtype=torch.int64,  device=dev),
        )
        self._static[key] = buf
        return buf

    def prepare_static_inputs(
        self,
        token_ids: list[int],
        positions: list[int],
        block_tables: list[list[int]],
        seq_lens: list[int],
        slot_mappings: list[int],
        batch_bucket: int,
        ctx_bucket: int,
    ) -> StaticBuffers:

        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        B_actual = len(token_ids)

        max_seq = max(seq_lens)
        if max_seq > ctx_bucket:
            raise ValueError(
                f"seq_len {max_seq} exceeds ctx_bucket {ctx_bucket} "
                f"(largest CTX_BUCKET is {max(CTX_BUCKETS)}). Reduce the "
                f"context length or the step count so that "
                f"ctx_len + decode_steps fits inside a bucket."
            )

        self._seq_lens_cpu = list(seq_lens) + [1] * (batch_bucket - B_actual)

        buf.input_ids[:B_actual].copy_(
            torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        )
        buf.position_ids[:B_actual].copy_(
            torch.tensor(positions, dtype=torch.int64, device=self.device)
        )
        buf.seq_lens[:B_actual].copy_(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        )
        buf.slot_mapping[:B_actual].copy_(
            torch.tensor(slot_mappings, dtype=torch.int64, device=self.device)
        )

        if B_actual < batch_bucket:
            buf.seq_lens[B_actual:].zero_()

        for i, bt in enumerate(block_tables):
            n = len(bt)
            buf.block_tables[i, :n].copy_(
                torch.tensor(bt, dtype=torch.int32, device=self.device)
            )
            buf.block_tables[i, n:].zero_()

        return buf

    @torch.inference_mode()
    def _forward_static(
        self,
        buf: StaticBuffers,
        batch_bucket: int,
        ctx_bucket: int,
    ) -> torch.Tensor:

        B = batch_bucket

        for m in self.model.modules():
            if isinstance(m, StaticDecodeAttention):
                m._ctx_bucket = ctx_bucket

        query_start_loc_cpu = list(range(B + 1))
        seq_lens_cpu = (self._seq_lens_cpu[:B] if len(self._seq_lens_cpu) >= B
                        else [1] * B)
        context_lens_cpu = [s - 1 for s in seq_lens_cpu]
        attn_metadata = AttentionMetadata(
            slot_mapping      = buf.slot_mapping[:B],
            block_tables      = buf.block_tables[:B],
            query_start_loc   = torch.arange(B + 1, dtype=torch.int32, device=self.device),
            seq_lens          = buf.seq_lens[:B],
            context_lens      = buf.seq_lens[:B] - 1,
            max_query_len     = 1,
            max_seq_len       = ctx_bucket,
            query_start_loc_cpu = query_start_loc_cpu,
            seq_lens_cpu      = seq_lens_cpu,
            context_lens_cpu  = context_lens_cpu,
        )

        with set_forward_context(attn_metadata):
            hidden = self.model(buf.input_ids[:B], buf.position_ids[:B])

        logits = self.model.compute_logits(hidden)
        buf.logits[:B].copy_(logits)

        buf.next_tokens[:B].copy_(logits.argmax(dim=-1))
        return buf.next_tokens

    def capture_one(self, batch_bucket: int, ctx_bucket: int, warmup_steps: int = 3):

        from graph_runner import CUDAGraphRunner

        key = (batch_bucket, ctx_bucket)
        if key in self._graphs:
            return self._graphs[key]

        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_steps):
                self._forward_static(buf, batch_bucket, ctx_bucket)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        runner = CUDAGraphRunner()
        runner.capture(self._forward_static, buf, batch_bucket, ctx_bucket,
                       pool=self._pool)
        self._graphs[key] = runner

        if self._pool is None:
            self._pool = runner.pool

        return runner

    def capture_all(self, warmup_steps: int = 3):

        mem_before = torch.cuda.memory_allocated()

        for cb in reversed(CTX_BUCKETS):
            for bb in reversed(BATCH_BUCKETS):
                self.capture_one(bb, cb, warmup_steps=warmup_steps)
                print(f"  Captured graph B={bb:2d} ctx={cb:4d}", flush=True)

        mem_after = torch.cuda.memory_allocated()
        print(f"  Graph pool total: {(mem_after - mem_before) / 1e9:.2f} GB for "
              f"{len(self._graphs)} graphs")

    def replay(self, buf: StaticBuffers, batch_bucket: int, ctx_bucket: int) -> torch.Tensor:

        key = (batch_bucket, ctx_bucket)
        if key not in self._graphs:
            raise RuntimeError(f"No graph captured for {key}. Call capture_all() first.")
        self._graphs[key].replay()
        return buf.next_tokens
