"""
Shared building blocks — RMSNorm, SwiGLU, Rotary Embeddings, TransformerBlock.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig


# ═════════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


# ═════════════════════════════════════════════════════════════════════════════
# SwiGLU Feed-Forward
# ═════════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: SiLU(xW₁) ⊙ xW₃ then W₂."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ═════════════════════════════════════════════════════════════════════════════
# Rotary Position Embeddings (RoPE)
# ═════════════════════════════════════════════════════════════════════════════

def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex-valued RoPE frequency tensor of shape (max_seq_len, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float() / scaling_factor
    angles = torch.outer(t, freqs)  # (seq_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim) — head_dim must be even.
        freqs: (seq_len, head_dim // 2) complex tensor from precompute_rope_frequencies.

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    # Reshape x into complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Broadcast freqs to (1, seq_len, 1, head_dim//2)
    freqs = freqs[None, :x_complex.shape[1], None, :]
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.type_as(x)


# ═════════════════════════════════════════════════════════════════════════════
# Standard Transformer Block (used in Prelude and Coda)
# ═════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with GQA attention and SwiGLU FFN.

    Used in the Prelude and Coda stages (non-looped, non-MoE).
    """

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.dim // cfg.n_heads
        n_kv = cfg.n_kv_heads if cfg.attn_type == "gqa" else cfg.n_heads

        # Attention
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, n_kv * head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * head_dim, cfg.dim, bias=False)
        self.attn_dropout_p = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()

        self.n_heads = cfg.n_heads
        self.n_kv_heads = n_kv
        self.head_dim = head_dim

        # FFN
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        ffn_hidden = int(cfg.dim * 8 / 3)  # standard SwiGLU sizing
        # Round up to nearest multiple of 64 for efficiency
        ffn_hidden = ((ffn_hidden + 63) // 64) * 64
        self.ffn = SwiGLU(cfg.dim, ffn_hidden, cfg.resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # ── Self-Attention ───────────────────────────────────────────────
        h = self.attn_norm(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(h).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(h).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        rope_slice = rope_freqs[:T, :self.head_dim // 2]
        q = apply_rotary_emb(q, rope_slice)
        k = apply_rotary_emb(k, rope_slice)

        # Expand KV heads for GQA
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, T, self.n_heads, self.head_dim)

        # (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=(mask is None)
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        attn = self.resid_dropout(self.wo(attn))

        x = x + attn

        # ── Feed-Forward ─────────────────────────────────────────────────
        x = x + self.ffn(self.ffn_norm(x))

        return x
