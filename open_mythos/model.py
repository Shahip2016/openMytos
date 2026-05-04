"""
OpenMythos — Recurrent-Depth Transformer with Mixture-of-Experts.

Main model class assembling all components:
    Embedding → Prelude → Recurrent Block (×K loops) → Coda → LM Head

The Recurrent Block uses weight-shared transformer layers with:
    • MLA or GQA attention
    • Sparse MoE feed-forward
    • LTI-stable recurrent injection
    • Depth-wise LoRA adapters
    • (Optional) Adaptive Computation Time halting
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from open_mythos.config import MythosConfig
from open_mythos.layers import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_frequencies,
)
from open_mythos.attention import MultiLatentAttention, GroupedQueryAttention
from open_mythos.moe import SparseMoEFFN
from open_mythos.recurrent import (
    LTIRecurrentGate,
    DepthLoRAAdapter,
    AdaptiveComputationTime,
)


# ═════════════════════════════════════════════════════════════════════════════
# Recurrent Transformer Block (the looped core)
# ═════════════════════════════════════════════════════════════════════════════

class RecurrentTransformerBlock(nn.Module):
    """Single recurrent block that is weight-shared across loop iterations.

    Architecture per step:
        1. Pre-norm → Attention (MLA or GQA)
        2. Pre-norm → Sparse MoE FFN
        3. LTI-stable state mixing with original Prelude output z₀
        4. Depth-wise LoRA adaptation (per loop step)
    """

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Loop-index sinusoidal embedding ───────────────────────────────
        # Encodes the current iteration index as a positional signal
        self.loop_embed = nn.Linear(cfg.dim, cfg.dim, bias=False)

        # ── Attention ─────────────────────────────────────────────────────
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        if cfg.attn_type == "mla":
            self.attn = MultiLatentAttention(cfg)
        else:
            self.attn = GroupedQueryAttention(cfg)

        # ── Sparse MoE FFN ────────────────────────────────────────────────
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.moe_ffn = SparseMoEFFN(cfg)

        # ── Recurrent stability ───────────────────────────────────────────
        self.lti_gate = LTIRecurrentGate(cfg.dim)
        self.lora_adapter = DepthLoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)

    def _sinusoidal_loop_embedding(
        self, step: int, dim: int, device: torch.device
    ) -> torch.Tensor:
        """Create sinusoidal embedding for the current loop iteration index."""
        pe = torch.zeros(dim, device=device)
        position = torch.tensor([step], device=device, dtype=torch.float32)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / dim)
        )
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe  # (dim,)

    def forward(
        self,
        h: torch.Tensor,
        z0: torch.Tensor,
        rope_freqs: torch.Tensor,
        step: int,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One iteration of the recurrent block.

        Args:
            h: Current hidden state (B, T, D)
            z0: Encoded input from Prelude, re-injected each step (B, T, D)
            rope_freqs: Precomputed RoPE frequencies
            step: Current loop iteration (0-indexed)
            mask: Optional attention mask

        Returns:
            h_new: Updated hidden state (B, T, D)
            aux_loss: MoE load-balancing loss (scalar)
        """
        D = h.shape[-1]

        # ── Inject loop-index embedding ───────────────────────────────────
        loop_pe = self._sinusoidal_loop_embedding(step, D, h.device)
        h_in = h + loop_pe.unsqueeze(0).unsqueeze(0)  # broadcast (1, 1, D)

        # ── Attention ─────────────────────────────────────────────────────
        normed = self.attn_norm(h_in)
        attn_out = self.attn(normed, rope_freqs, mask)
        h_in = h_in + attn_out

        # ── MoE FFN ──────────────────────────────────────────────────────
        normed = self.ffn_norm(h_in)
        moe_out, aux_loss = self.moe_ffn(normed)
        update = h_in + moe_out

        # ── LTI-Stable Recurrent Injection ────────────────────────────────
        h_new = self.lti_gate(h, update, z0)

        # ── Depth-Wise LoRA Adapter ───────────────────────────────────────
        h_new = self.lora_adapter(h_new, step)

        return h_new, aux_loss


# ═════════════════════════════════════════════════════════════════════════════
# OpenMythos — Full Model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenMythosOutput:
    """Container for model outputs."""
    logits: torch.Tensor           # (B, T, vocab_size)
    aux_loss: torch.Tensor         # scalar — MoE load-balancing loss
    ponder_cost: torch.Tensor      # scalar — ACT ponder cost (0 if ACT disabled)


class OpenMythos(nn.Module):
    """Recurrent-Depth Transformer with Mixture-of-Experts.

    Architecture:
        Token Embedding (weight-tied with LM head)
        → Prelude: standard transformer blocks (run once)
        → Recurrent Block: looped K times with shared weights
            — MLA or GQA Attention
            — Sparse MoE FFN
            — LTI-stable recurrent injection
            — Depth-wise LoRA adapters
            — (Optional) ACT halting
        → Coda: standard transformer blocks (run once)
        → RMSNorm → LM Head
    """

    def __init__(self, cfg: MythosConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Token Embedding ───────────────────────────────────────────────
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)

        # ── Prelude (standard transformer blocks) ─────────────────────────
        self.prelude = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.prelude_layers)
        ])

        # ── Recurrent Block (weight-shared, looped) ──────────────────────
        self.recurrent_block = RecurrentTransformerBlock(cfg)

        # ── Coda (standard transformer blocks) ───────────────────────────
        self.coda = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.coda_layers)
        ])

        # ── Output ───────────────────────────────────────────────────────
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying: LM head optionally shares weights with token embedding
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # ── Adaptive Computation Time (optional) ─────────────────────────
        self.act: Optional[AdaptiveComputationTime] = None
        if cfg.use_act:
            self.act = AdaptiveComputationTime(cfg.dim, cfg.act_threshold)

        # ── Precompute RoPE frequencies ──────────────────────────────────
        # Use max of qk_rope_dim (MLA) and head_dim (GQA/Prelude/Coda)
        rope_dim = max(cfg.dim // cfg.n_heads, cfg.qk_head_dim)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(rope_dim, cfg.max_seq_len, cfg.rope_theta, cfg.rope_scaling_factor),
            persistent=False,
        )

        # ── Initialisation ───────────────────────────────────────────────
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise weights with standard deviation scaled by dim."""
        if isinstance(module, nn.Linear):
            std = self.cfg.init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> OpenMythosOutput:
        """Forward pass through the full model.

        Args:
            input_ids: (B, T) token indices
            mask: Optional attention mask

        Returns:
            OpenMythosOutput with logits, aux_loss, and ponder_cost
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Determine attention mask
        if mask is None:
            if self.cfg.sliding_window is not None:
                # Causal sliding window mask
                mask = torch.ones(T, T, dtype=torch.bool, device=device)
                mask = torch.tril(mask) & torch.triu(mask, diagonal=-self.cfg.sliding_window + 1)
                # Convert to attention mask format
                mask = torch.zeros(T, T, device=device).masked_fill(~mask, float("-inf"))
            else:
                # is_causal handles standard causal mask internally
                pass

        # ── Token Embedding ───────────────────────────────────────────────
        x = self.tok_emb(input_ids)
        # Scale embeddings
        x = x * math.sqrt(self.cfg.dim)

        # ── Prelude ──────────────────────────────────────────────────────
        for block in self.prelude:
            if self.cfg.use_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, self.rope_freqs, mask, use_reentrant=False)
            else:
                x = block(x, self.rope_freqs, mask)

        # Save encoded input for LTI re-injection
        z0 = x

        # ── Recurrent Block (looped) ─────────────────────────────────────
        h = x
        total_aux_loss = torch.tensor(0.0, device=device)
        intermediate_states: list[torch.Tensor] = []

        for step in range(self.cfg.max_loop_iters):
            if self.cfg.use_checkpointing and self.training:
                h, aux_loss = checkpoint.checkpoint(
                    self.recurrent_block, h, z0, self.rope_freqs, step, mask, use_reentrant=False
                )
            else:
                h, aux_loss = self.recurrent_block(h, z0, self.rope_freqs, step, mask)
            total_aux_loss = total_aux_loss + aux_loss
            if self.act is not None:
                intermediate_states.append(h)

        # ── ACT halting (optional) ───────────────────────────────────────
        ponder_cost = torch.tensor(0.0, device=device)
        if self.act is not None and len(intermediate_states) > 0:
            h, ponder_cost = self.act(intermediate_states)

        # Average aux_loss across loop steps
        total_aux_loss = total_aux_loss / self.cfg.max_loop_iters

        # ── Coda ─────────────────────────────────────────────────────────
        x = h
        for block in self.coda:
            if self.cfg.use_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, self.rope_freqs, mask, use_reentrant=False)
            else:
                x = block(x, self.rope_freqs, mask)

        # ── Output ───────────────────────────────────────────────────────
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if self.cfg.final_logit_softcapping is not None:
            logits = logits / self.cfg.final_logit_softcapping
            logits = torch.tanh(logits) * self.cfg.final_logit_softcapping

        return OpenMythosOutput(
            logits=logits,
            aux_loss=total_aux_loss,
            ponder_cost=ponder_cost,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Args:
            input_ids: (B, T) prompt token indices
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)

        Returns:
            (B, T + max_new_tokens) tensor of generated token indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if necessary
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]

            # Forward pass
            output = self.forward(idx_cond)
            logits = output.logits[:, -1, :] / max(temperature, 1e-8)

            # ── Repetition Penalty ───────────────────────────────────────
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, input_ids)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, input_ids, score)

            # ── Min-p filtering ──────────────────────────────────────────
            if min_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1, keepdim=True).values
                indices_to_remove = probs < (min_p * max_probs)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # ── Top-k filtering ──────────────────────────────────────────
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            # ── Top-p (nucleus) filtering ────────────────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cum_probs > top_p
                # Shift right so the first token above threshold is kept
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # ── Sample ───────────────────────────────────────────────────
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        """Autoregressive token generation yielding tokens sequentially."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            output = self.forward(idx_cond)
            logits = output.logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, input_ids)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, input_ids, score)

            if min_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1, keepdim=True).values
                indices_to_remove = probs < (min_p * max_probs)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k_val, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            yield next_token
            input_ids = torch.cat([input_ids, next_token], dim=1)

    @torch.no_grad()
    def calculate_perplexity(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> float:
        """Calculate perplexity of the model on the given input_ids.
        
        If labels is None, input_ids are used as labels (shifted internally).
        """
        self.eval()
        if labels is None:
            labels = input_ids

        output = self.forward(input_ids)
        logits = output.logits
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return math.exp(loss.item())

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        prelude_params = sum(_count(b) for b in self.prelude)
        recurrent_params = _count(self.recurrent_block)
        coda_params = sum(_count(b) for b in self.coda)
        embedding_params = self.tok_emb.weight.numel()
        head_params = 0  # weight-tied with embedding
        norm_params = _count(self.final_norm)
        act_params = _count(self.act) if self.act is not None else 0

        total = sum(p.numel() for p in self.parameters())

        return {
            "embedding": embedding_params,
            "prelude": prelude_params,
            "recurrent_block": recurrent_params,
            "coda": coda_params,
            "final_norm": norm_params,
            "act": act_params,
            "lm_head": head_params,
            "total": total,
        }
