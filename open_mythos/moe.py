"""
Sparse Mixture-of-Experts FFN — DeepSeekMoE-inspired.

Features:
  • Fine-grained routed experts with top-k gating
  • Always-active shared experts
  • Auxiliary load-balancing loss to prevent expert collapse
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig
from open_mythos.layers import SwiGLU


class MoERouter(nn.Module):
    """Top-k gating router for sparse expert selection."""

    def __init__(self, dim: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(dim, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq_len, dim)

        Returns:
            weights: (batch * seq_len, top_k) — softmax weights for selected experts
            indices: (batch * seq_len, top_k) — expert indices
            aux_loss: scalar — load-balancing auxiliary loss
        """
        logits = self.gate(x)  # (N, n_experts)
        scores = F.softmax(logits, dim=-1)

        # Top-k selection
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        # Renormalise selected weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        # ── Load-balancing loss (Switch Transformer style) ───────────────
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability for expert i
        # loss = n_experts * sum(f_i * P_i)
        N = x.shape[0]
        # One-hot assignment for each selected expert
        mask = F.one_hot(indices, self.n_experts).float()  # (N, top_k, E)
        mask = mask.sum(dim=1)  # (N, E) — could be >1 if same expert picked twice
        f = mask.mean(dim=0)   # (E,) fraction of tokens per expert
        P = scores.mean(dim=0) # (E,) mean probability per expert
        aux_loss = (self.n_experts * (f * P).sum())

        return weights, indices, aux_loss


class SparseMoEFFN(nn.Module):
    """Sparse Mixture-of-Experts feed-forward network.

    Combines routed experts (top-k selection) with always-active shared experts.
    """

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.n_shared_experts = cfg.n_shared_experts
        self.top_k = cfg.n_experts_per_tok

        # Router
        self.router = MoERouter(cfg.dim, cfg.n_experts, cfg.n_experts_per_tok)

        # Routed experts — each is a small SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLU(cfg.dim, cfg.expert_dim, cfg.resid_dropout, cfg.hidden_act)
            for _ in range(cfg.n_experts)
        ])

        # Shared experts — always active, outputs added unconditionally
        self.shared_experts = nn.ModuleList([
            SwiGLU(cfg.dim, cfg.expert_dim, cfg.resid_dropout, cfg.hidden_act)
            for _ in range(cfg.n_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            output: (batch, seq_len, dim)
            aux_loss: scalar load-balancing loss
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (N, D)

        # ── Routing ──────────────────────────────────────────────────────
        weights, indices, aux_loss = self.router(x_flat)
        # weights: (N, top_k), indices: (N, top_k)

        # ── Routed expert computation ────────────────────────────────────
        out = torch.zeros_like(x_flat)
        capacity = None
        if self.cfg.expert_capacity_factor is not None and self.training:
            capacity = int(self.cfg.expert_capacity_factor * (B * T * self.top_k) / self.n_experts)

        for i, expert in enumerate(self.experts):
            # Mask of (token, slot) pairs routed to expert i
            expert_mask = (indices == i)  # (N, top_k) bool
            if not expert_mask.any():
                continue
            
            # Weight: sum of weights across slots where this expert was selected
            w = (weights * expert_mask.float()).sum(dim=-1)  # (N,)
            
            # Get token indices that route to this expert (any slot)
            token_mask = expert_mask.any(dim=-1)  # (N,) bool
            
            # Capacity limiting
            if capacity is not None and token_mask.sum().item() > capacity:
                # Get the indices of tokens routed to this expert
                routed_indices = token_mask.nonzero().squeeze(-1)
                # Sort them by router weight
                routed_weights = w[routed_indices]
                _, top_idx = torch.topk(routed_weights, capacity)
                # Keep only the top 'capacity' tokens
                new_token_mask = torch.zeros_like(token_mask)
                new_token_mask[routed_indices[top_idx]] = True
                token_mask = new_token_mask

            expert_input = x_flat[token_mask]  # (n_tokens, D)
            expert_output = expert(expert_input)  # (n_tokens, D)
            w_selected = w[token_mask].unsqueeze(-1)  # (n_tokens, 1)
            out[token_mask] += expert_output * w_selected

        # ── Shared experts (always active) ───────────────────────────────
        for shared_expert in self.shared_experts:
            out = out + shared_expert(x_flat)

        return out.view(B, T, D), aux_loss
