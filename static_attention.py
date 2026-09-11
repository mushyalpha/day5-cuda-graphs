from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_kv_batched(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_bucket: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    n_blocks_needed = (ctx_bucket + block_size - 1) // block_size
    B = block_tables.shape[0]
    n_kv_heads = kv_cache.shape[3]
    head_dim   = kv_cache.shape[4]

    bt = block_tables[:, :n_blocks_needed]
    bt_flat = bt.reshape(-1)

    k_blocks = kv_cache[0][bt_flat]
    v_blocks = kv_cache[1][bt_flat]

    k = k_blocks.reshape(B, n_blocks_needed * block_size, n_kv_heads, head_dim)
    v = v_blocks.reshape(B, n_blocks_needed * block_size, n_kv_heads, head_dim)

    k = k[:, :ctx_bucket].permute(0, 2, 1, 3)
    v = v[:, :ctx_bucket].permute(0, 2, 1, 3)
    return k, v

def make_decode_mask(
    seq_lens: torch.Tensor,
    ctx_bucket: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    B = seq_lens.shape[0]
    positions = torch.arange(ctx_bucket, device=device)
    seq_lens_long = seq_lens.to(torch.int64)
    valid = positions[None, :] < seq_lens_long[:, None]

    additive = torch.zeros(B, 1, 1, ctx_bucket, dtype=dtype, device=device)
    additive = additive.masked_fill(~valid[:, None, None, :], float("-inf"))
    return additive

class StaticDecodeAttention(nn.Module):

    _ctx_bucket: int = 256
    _block_size: int = 16

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        from babyvllm.worker.context import get_forward_context
        from babyvllm.layers.attention import store_kvcache

        ctx = get_forward_context()
        if self.kv_cache is None or ctx is None:

            return _naive_causal_attention(q, k, v, self.scale, self.num_queries_per_kv)

        md = ctx.attn_metadata
        B = md.seq_lens.shape[0]
        ctx_bucket = self._ctx_bucket
        block_size = self._block_size

        store_kvcache(k, v, self.kv_cache, md.slot_mapping)

        K, V = gather_kv_batched(self.kv_cache, md.block_tables, ctx_bucket, block_size)

        Q = q.reshape(B, self.num_heads, 1, self.head_dim)

        mask = make_decode_mask(md.seq_lens, ctx_bucket, q.device, q.dtype)

        # Fold the GQA group into the query-length dim. Passing enable_gqa=True
        # plus an additive mask selects the math backend, which implements GQA
        # as key/value.repeat_interleave — ~105 GB/step at B=64. This view is
        # mathematically identical and never expands K or V.
        group = self.num_queries_per_kv
        if group > 1:
            Q = Q.reshape(B, self.num_kv_heads, group, self.head_dim)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            scale=self.scale,
        )
        if group > 1:
            out = out.reshape(B, self.num_heads, 1, self.head_dim)

        return out.squeeze(2).reshape(B, self.num_heads, self.head_dim)

def _naive_causal_attention(q, k, v, scale, num_queries_per_kv):
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        is_causal=True, scale=scale,
    )
    return o.transpose(0, 1)
