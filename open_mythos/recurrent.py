"""
Recurrent stability mechanisms for the looped transformer block.

• LTI-Stable Recurrent Injection — prevents residual explosion across loop steps
• Depth-Wise LoRA Adapters — per-iteration low-rank adaptation
• Adaptive Computation Time (ACT) — dynamic halting
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_mythos.config import MythosConfig


# ═════════════════════════════════════════════════════════════════════════════
# LTI-Stable Recurrent Injection
# ═════════════════════════════════════════════════════════════════════════════

class LTIRecurrentGate(nn.Module):
    """Linear Time-Invariant stable recurrent injection gate.

    At each loop iteration, the hidden state is updated as:
        h_{t+1} = α · h_t + (1 - α) · update
    where α ∈ (0, 1) is a learned per-dimension decay factor clamped via sigmoid.

    This ensures the recurrent dynamics are contractive (eigenvalues < 1),
    preventing the residual stream from exploding across many loop iterations.

    Inspired by the Parcae architecture's stability constraints.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Learned pre-sigmoid decay — initialised near 0 so α ≈ 0.5
        self.log_alpha = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        h: torch.Tensor,
        update: torch.Tensor,
        z0: torch.Tensor,
    ) -> torch.Tensor:
        """Stable state update with input re-injection.

        Args:
            h: Current hidden state (B, T, D)
            update: New update from the recurrent block (B, T, D)
            z0: Original encoded input from Prelude, re-injected for grounding

        Returns:
            New hidden state (B, T, D)
        """
        alpha = torch.sigmoid(self.log_alpha)  # (D,) in [0, 1]
        # Mix: decay old state, blend in new update + re-inject encoded input
        h_new = alpha * h + (1.0 - alpha) * (update + z0)
        return h_new


# ═════════════════════════════════════════════════════════════════════════════
# Depth-Wise LoRA Adapter
# ═════════════════════════════════════════════════════════════════════════════

class DepthLoRAAdapter(nn.Module):
    """Per-iteration low-rank adaptation (LoRA) for the recurrent block.

    Each loop iteration applies a distinct rank-r adapter so that the
    shared weights can behave differently at each depth without adding
    a full parameter set per iteration.

    Parameters scale as: max_loop_iters × 2 × dim × rank
    (much cheaper than full per-depth copies).
    """

    def __init__(self, dim: int, rank: int, max_iters: int) -> None:
        super().__init__()
        self.max_iters = max_iters
        self.rank = rank
        # Down projections: one per iteration
        self.down = nn.Parameter(torch.randn(max_iters, dim, rank) * 0.01)
        # Up projections: one per iteration
        self.up = nn.Parameter(torch.randn(max_iters, rank, dim) * 0.01)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Apply the LoRA adapter for a specific loop step.

        Args:
            x: (B, T, D)
            step: Current loop iteration index (0-indexed)

        Returns:
            x + LoRA(x) of shape (B, T, D)
        """
        step = min(step, self.max_iters - 1)
        # x @ down[step] @ up[step]  →  (B, T, D) @ (D, R) @ (R, D) = (B, T, D)
        delta = F.linear(F.linear(x, self.down[step].T), self.up[step].T)
        return x + delta


# ═════════════════════════════════════════════════════════════════════════════
# Adaptive Computation Time (ACT)
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time module (Graves, 2016).

    Learns a halting probability at each loop iteration for each token.
    When the cumulative halting probability exceeds the threshold, the
    token stops being updated. Produces a *ponder cost* added to the loss
    to encourage efficient computation.

    The final hidden state is a weighted combination of intermediate states
    using the halting probabilities and remainder.
    """

    def __init__(self, dim: int, threshold: float = 0.99) -> None:
        super().__init__()
        self.threshold = threshold
        self.halt_proj = nn.Linear(dim, 1, bias=True)
        # Initialise bias so halting starts low (model loops by default)
        nn.init.constant_(self.halt_proj.bias, -3.0)

    def forward(
        self,
        states: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ACT-weighted output from a list of intermediate states.

        Args:
            states: List of K tensors, each (B, T, D), one per loop step.

        Returns:
            output: (B, T, D) — weighted combination of states
            ponder_cost: scalar — expected number of steps (for loss)
        """
        K = len(states)
        B, T, D = states[0].shape
        device = states[0].device

        # Accumulators
        halting_prob = torch.zeros(B, T, 1, device=device)
        remainders = torch.zeros(B, T, 1, device=device)
        n_updates = torch.zeros(B, T, 1, device=device)
        output = torch.zeros(B, T, D, device=device)

        for i, state in enumerate(states):
            p = torch.sigmoid(self.halt_proj(state))  # (B, T, 1)

            still_running = (halting_prob < 1.0).float()

            # On last step, use remainder
            if i == K - 1:
                p = 1.0 - halting_prob
            else:
                # Clamp so we don't exceed 1.0
                p = torch.min(p, 1.0 - halting_prob)

            halting_prob = halting_prob + p * still_running
            output = output + p * still_running * state
            n_updates = n_updates + still_running
            remainders = remainders + p * still_running

        # Ponder cost: mean number of steps taken (differentiable proxy)
        ponder_cost = n_updates.mean()

        return output, ponder_cost
