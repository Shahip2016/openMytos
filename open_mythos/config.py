"""
MythosConfig — All architecture hyperparameters for OpenMythos.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MythosConfig:
    """Configuration for the OpenMythos Recurrent-Depth Transformer.

    The architecture follows a three-part structure:
        Prelude (standard layers) → Recurrent Block (looped) → Coda (standard layers)

    The recurrent block uses sparse MoE feed-forward layers with either
    Multi-Latent Attention (MLA) or Grouped Query Attention (GQA).
    """

    # ── Vocabulary & Dimensions ──────────────────────────────────────────
    vocab_size: int = 32_000
    dim: int = 2048
    max_seq_len: int = 4096
    tie_word_embeddings: bool = True

    # ── Structural Depth ─────────────────────────────────────────────────
    prelude_layers: int = 4
    coda_layers: int = 4
    max_loop_iters: int = 8
    use_checkpointing: bool = False

    # ── Attention ────────────────────────────────────────────────────────
    attn_type: Literal["mla", "gqa"] = "mla"
    n_heads: int = 16
    # GQA-specific
    n_kv_heads: int = 4
    # MLA-specific (DeepSeek-V2 style latent compression)
    kv_lora_rank: int = 64
    q_lora_rank: int = 128
    qk_head_dim: int = 64
    v_head_dim: int = 64
    qk_rope_dim: int = 32  # portion of qk_head_dim that receives RoPE
    sliding_window: Optional[int] = None

    # ── Mixture-of-Experts ───────────────────────────────────────────────
    n_experts: int = 8
    n_shared_experts: int = 2
    n_experts_per_tok: int = 2
    expert_dim: int = 1024

    # ── Recurrent Stability ──────────────────────────────────────────────
    lora_rank: int = 16          # depth-wise LoRA adapter rank
    use_act: bool = False        # enable Adaptive Computation Time
    act_threshold: float = 0.99  # halting threshold for ACT

    # ── Regularisation & Norms ───────────────────────────────────────────
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    init_std: float = 0.02

    # ── MoE Auxiliary Loss ───────────────────────────────────────────────
    moe_aux_loss_weight: float = 0.01

    # ── Logit Softcapping ────────────────────────────────────────────────
    final_logit_softcapping: Optional[float] = None

    def __post_init__(self) -> None:
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.attn_type in ("mla", "gqa"), (
            f"attn_type must be 'mla' or 'gqa', got '{self.attn_type}'"
        )
        if self.attn_type == "gqa":
            assert self.n_heads % self.n_kv_heads == 0, (
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        assert self.n_experts_per_tok <= self.n_experts, (
            f"n_experts_per_tok ({self.n_experts_per_tok}) must be "
            f"<= n_experts ({self.n_experts})"
        )
        assert self.qk_rope_dim <= self.qk_head_dim, (
            f"qk_rope_dim ({self.qk_rope_dim}) must be "
            f"<= qk_head_dim ({self.qk_head_dim})"
        )

    @property
    def head_dim(self) -> int:
        """Per-head dimension for GQA attention."""
        return self.dim // self.n_heads
