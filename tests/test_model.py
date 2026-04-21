"""
Tests for the OpenMythos Recurrent-Depth Transformer.
"""

import pytest
import torch

from open_mythos import OpenMythos, MythosConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg_mla():
    """Small MLA config that fits on CPU."""
    return MythosConfig(
        vocab_size=500,
        dim=128,
        max_seq_len=256,
        prelude_layers=1,
        coda_layers=1,
        max_loop_iters=3,
        n_heads=4,
        n_kv_heads=2,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_head_dim=32,
        v_head_dim=32,
        qk_rope_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=4,
    )


@pytest.fixture
def small_cfg_gqa():
    """Small GQA config that fits on CPU."""
    return MythosConfig(
        vocab_size=500,
        dim=128,
        max_seq_len=256,
        prelude_layers=1,
        coda_layers=1,
        max_loop_iters=3,
        n_heads=4,
        n_kv_heads=2,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=4,
    )


@pytest.fixture
def small_cfg_act(small_cfg_mla):
    """MLA config with ACT enabled."""
    return MythosConfig(
        vocab_size=small_cfg_mla.vocab_size,
        dim=small_cfg_mla.dim,
        max_seq_len=small_cfg_mla.max_seq_len,
        prelude_layers=small_cfg_mla.prelude_layers,
        coda_layers=small_cfg_mla.coda_layers,
        max_loop_iters=small_cfg_mla.max_loop_iters,
        n_heads=small_cfg_mla.n_heads,
        n_kv_heads=small_cfg_mla.n_kv_heads,
        attn_type="mla",
        kv_lora_rank=small_cfg_mla.kv_lora_rank,
        q_lora_rank=small_cfg_mla.q_lora_rank,
        qk_head_dim=small_cfg_mla.qk_head_dim,
        v_head_dim=small_cfg_mla.v_head_dim,
        qk_rope_dim=small_cfg_mla.qk_rope_dim,
        n_experts=small_cfg_mla.n_experts,
        n_shared_experts=small_cfg_mla.n_shared_experts,
        n_experts_per_tok=small_cfg_mla.n_experts_per_tok,
        expert_dim=small_cfg_mla.expert_dim,
        lora_rank=small_cfg_mla.lora_rank,
        use_act=True,
        act_threshold=0.99,
    )


# ── Forward Pass Tests ───────────────────────────────────────────────────────

class TestForwardPass:
    """Test forward pass shape correctness."""

    def test_mla_forward_shape(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        B, T = 2, 32
        x = torch.randint(0, small_cfg_mla.vocab_size, (B, T))
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (B, T, small_cfg_mla.vocab_size)

    def test_gqa_forward_shape(self, small_cfg_gqa):
        model = OpenMythos(small_cfg_gqa)
        model.eval()
        B, T = 2, 32
        x = torch.randint(0, small_cfg_gqa.vocab_size, (B, T))
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (B, T, small_cfg_gqa.vocab_size)

    def test_act_forward_shape(self, small_cfg_act):
        model = OpenMythos(small_cfg_act)
        model.eval()
        B, T = 2, 16
        x = torch.randint(0, small_cfg_act.vocab_size, (B, T))
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (B, T, small_cfg_act.vocab_size)
        assert out.ponder_cost.item() > 0, "ACT ponder cost should be positive"

    def test_single_token_input(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        x = torch.randint(0, small_cfg_mla.vocab_size, (1, 1))
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (1, 1, small_cfg_mla.vocab_size)


# ── Auxiliary Loss Tests ─────────────────────────────────────────────────────

class TestAuxLoss:
    """Test that auxiliary losses are finite and non-negative."""

    def test_aux_loss_finite(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        x = torch.randint(0, small_cfg_mla.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert out.aux_loss.isfinite(), "Aux loss must be finite"
        assert out.aux_loss.item() >= 0, "Aux loss must be non-negative"

    def test_ponder_cost_zero_without_act(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        x = torch.randint(0, small_cfg_mla.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert out.ponder_cost.item() == 0.0, \
            "Ponder cost should be 0 when ACT is disabled"


# ── Generation Tests ─────────────────────────────────────────────────────────

class TestGeneration:
    """Test autoregressive generation."""

    def test_generate_length(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        prompt = torch.randint(0, small_cfg_mla.vocab_size, (1, 4))
        max_new = 8
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=max_new)
        assert generated.shape == (1, 4 + max_new)

    def test_generate_valid_tokens(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        prompt = torch.randint(0, small_cfg_mla.vocab_size, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=8)
        assert (generated >= 0).all()
        assert (generated < small_cfg_mla.vocab_size).all()

    def test_generate_preserves_prompt(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.eval()
        prompt = torch.randint(0, small_cfg_mla.vocab_size, (1, 4))
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=8)
        assert torch.equal(generated[:, :4], prompt)


# ── Config Validation Tests ──────────────────────────────────────────────────

class TestConfigValidation:
    """Test config validation catches bad values."""

    def test_dim_not_divisible_by_heads(self):
        with pytest.raises(AssertionError, match="divisible by n_heads"):
            MythosConfig(dim=100, n_heads=7)

    def test_invalid_attn_type(self):
        with pytest.raises(AssertionError, match="attn_type must be"):
            MythosConfig(attn_type="invalid")

    def test_too_many_experts_per_tok(self):
        with pytest.raises(AssertionError, match="n_experts_per_tok"):
            MythosConfig(n_experts=4, n_experts_per_tok=8)


# ── Parameter Count Tests ────────────────────────────────────────────────────

class TestParameters:
    """Test parameter counting."""

    def test_count_parameters(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["embedding"] > 0
        assert counts["prelude"] > 0
        assert counts["recurrent_block"] > 0
        assert counts["coda"] > 0

    def test_weight_tying(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        assert model.lm_head.weight is model.tok_emb.weight, \
            "LM head and embedding should share weights"


# ── Gradient Tests ───────────────────────────────────────────────────────────

class TestGradients:
    """Test that gradients flow through the full model."""

    def test_gradients_flow(self, small_cfg_mla):
        model = OpenMythos(small_cfg_mla)
        model.train()
        x = torch.randint(0, small_cfg_mla.vocab_size, (1, 8))
        out = model(x)
        loss = out.logits.sum() + out.aux_loss
        loss.backward()
        # Check some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Gradients should flow through the model"
