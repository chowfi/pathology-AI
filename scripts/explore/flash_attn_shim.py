"""Drop-in PyTorch replacement for `flash_attn.flash_attn_varlen_func` (non-causal path only).

Prism2's perceiver imports flash-attn for one operation: cross-attention from a fixed set of latent queries
to a variable number of tile embeddings per slide, packed as flat "varlen" tensors. This shim computes the same
attention with torch's scaled_dot_product_attention, one sequence at a time.

Conventions matched to flash-attn:
  * q: (total_q, Hq, D), k/v: (total_k, Hkv, D), cu_seqlens_* are int32 prefix sums of per-sequence lengths.
  * Grouped-query attention: query head h uses kv head h // (Hq // Hkv)  (== repeat_interleave on kv heads).
  * Default softmax scale 1/sqrt(D).

Usage: import flash_attn_shim; flash_attn_shim.install()   # before loading Prism2
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    if causal:
        raise NotImplementedError("shim only supports causal=False (all Prism2 perceiver calls)")
    if dropout_p:
        raise NotImplementedError("shim is inference-only (dropout_p must be 0)")
    hq, hkv = q.shape[1], k.shape[1]
    out = torch.empty_like(q)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].transpose(0, 1)            # (Hq, Lq, D)
        ki = k[ck[i]:ck[i + 1]].transpose(0, 1)            # (Hkv, Lk, D)
        vi = v[ck[i]:ck[i + 1]].transpose(0, 1)
        if hq != hkv:
            ki = ki.repeat_interleave(hq // hkv, dim=0)
            vi = vi.repeat_interleave(hq // hkv, dim=0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
        out[cq[i]:cq[i + 1]] = oi.transpose(0, 1)
    return out


def install() -> None:
    """Register a fake `flash_attn` module so `from flash_attn import flash_attn_varlen_func` works.

    Import transformers' modeling utilities first, so its own "is flash-attn installed?" checks run against the
    real environment (answer: no) and it never routes its own layers to flash-attn.
    """
    import transformers.modeling_utils  # noqa: F401  (must run before the fake module exists)

    mod = types.ModuleType("flash_attn")
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    mod.__version__ = "shim"
    sys.modules["flash_attn"] = mod
