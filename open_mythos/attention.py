"""
Attention mechanisms — Multi-Latent Attention (MLA) and Grouped Query Attention (GQA).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.layers import RMSNorm, apply_rotary_emb


# ═════════════════════════════════════════════════════════════════════════════
# Multi-Latent Attention (MLA) — DeepSeek-V2 Style
# ═════════════════════════════════════════════════════════════════════════════

class MultiLatentAttention(nn.Module):
    """DeepSeek-V2 style Multi-Latent Attention.

    Key idea: compress K, V into a shared low-rank latent vector, then
    project back out. This reduces KV cache from O(n_heads * head_dim)
    to O(kv_lora_rank) per token.

    Additionally uses decoupled RoPE: a separate set of key heads receive
    positional embeddings and are concatenated with the content keys.
    """

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_head_dim = cfg.qk_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_rope_dim = cfg.qk_rope_dim
        self.qk_nope_dim = cfg.qk_head_dim - cfg.qk_rope_dim
        self.kv_lora_rank = cfg.kv_lora_rank

        # ── Query path ───────────────────────────────────────────────────
        # Compress then expand: dim → q_lora_rank → n_heads * (qk_head_dim + v_head_dim ... no)
        # Actually: dim → q_lora_rank is the compression, then project to Q heads
        self.q_compress = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        # From compressed, project to content Q (nope) and RoPE Q (rope) per head
        self.q_proj = nn.Linear(
            cfg.q_lora_rank,
            cfg.n_heads * cfg.qk_head_dim,
            bias=False,
        )

        # ── KV path (shared latent) ─────────────────────────────────────
        self.kv_compress = nn.Linear(cfg.dim, cfg.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        # From latent, project to content K and V per head
        self.k_proj = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * self.qk_nope_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * cfg.v_head_dim,
            bias=False,
        )

        # ── Decoupled RoPE keys ─────────────────────────────────────────
        # Separate projection for the RoPE portion of keys
        self.k_rope_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.qk_rope_dim, bias=False)

        # ── Output projection ───────────────────────────────────────────
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_dropout_p = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()
        self.cfg = cfg
        if cfg.qk_norm:
            self.qk_norm_q = RMSNorm(cfg.qk_head_dim, cfg.norm_eps)
            self.qk_norm_k = RMSNorm(cfg.qk_head_dim, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # ── Query ────────────────────────────────────────────────────────
        q = self.q_proj(self.q_norm(self.q_compress(x)))
        q = q.view(B, T, self.n_heads, self.qk_head_dim)
        if self.cfg.qk_norm:
            q = self.qk_norm_q(q)
        # Split into content (nope) and positional (rope) parts
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # ── KV (latent compression) ─────────────────────────────────────
        kv_latent = self.kv_norm(self.kv_compress(x))
        k_nope = self.k_proj(kv_latent).view(B, T, self.n_heads, self.qk_nope_dim)
        v = self.v_proj(kv_latent).view(B, T, self.n_heads, self.v_head_dim)

        # ── Decoupled RoPE keys ─────────────────────────────────────────
        k_rope = self.k_rope_proj(x).view(B, T, self.n_heads, self.qk_rope_dim)

        k_full = torch.cat([k_nope, k_rope], dim=-1)
        if self.cfg.qk_norm:
            k_full = self.qk_norm_k(k_full)
        k_nope, k_rope = k_full.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Apply RoPE to the positional portions only
        # Need freqs for qk_rope_dim // 2 complex pairs
        rope_freqs_slice = rope_freqs[:T, :self.qk_rope_dim // 2]
        q_rope = apply_rotary_emb(q_rope, rope_freqs_slice)
        k_rope = apply_rotary_emb(k_rope, rope_freqs_slice)

        # ── Concatenate content + positional ─────────────────────────────
        q_full = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, qk_head_dim)
        k_full = torch.cat([k_nope, k_rope], dim=-1)  # (B, T, H, qk_head_dim)

        # ── Attention ────────────────────────────────────────────────────
        q_full = q_full.transpose(1, 2)  # (B, H, T, qk_head_dim)
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)            # (B, H, T, v_head_dim)

        out = F.scaled_dot_product_attention(
            q_full, k_full, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=(mask is None),
            scale=1.0 / math.sqrt(self.qk_head_dim)
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.wo(out))


# ═════════════════════════════════════════════════════════════════════════════
# Grouped Query Attention (GQA)
# ═════════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """Standard GQA with fewer KV heads shared across query heads."""

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.attn_dropout_p = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()
        self.cfg = cfg
        if cfg.qk_norm:
            self.qk_norm_q = RMSNorm(self.head_dim, cfg.norm_eps)
            self.qk_norm_k = RMSNorm(self.head_dim, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.cfg.qk_norm:
            q = self.qk_norm_q(q)
            k = self.qk_norm_k(k)

        # Apply RoPE
        rope_slice = rope_freqs[:T, :self.head_dim // 2]
        q = apply_rotary_emb(q, rope_slice)
        k = apply_rotary_emb(k, rope_slice)

        # Expand KV for GQA
        if self.kv_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.kv_rep, -1)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.kv_rep, -1)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout_p if self.training else 0.0, is_causal=(mask is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.wo(out))
