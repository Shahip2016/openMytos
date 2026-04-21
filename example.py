"""
OpenMythos — Example Usage

Demonstrates model creation, forward pass, and generation with both
MLA and GQA attention variants.
"""

import torch
from open_mythos import OpenMythos, MythosConfig


def print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def demo_forward_pass(cfg: MythosConfig, label: str) -> None:
    """Run a forward pass and print diagnostics."""
    print_separator(f"Forward Pass — {label}")

    model = OpenMythos(cfg)
    model.eval()

    # Count parameters
    param_counts = model.count_parameters()
    print(f"Configuration:")
    print(f"  attn_type      = {cfg.attn_type}")
    print(f"  dim            = {cfg.dim}")
    print(f"  n_heads        = {cfg.n_heads}")
    print(f"  prelude_layers = {cfg.prelude_layers}")
    print(f"  coda_layers    = {cfg.coda_layers}")
    print(f"  max_loop_iters = {cfg.max_loop_iters}")
    print(f"  n_experts      = {cfg.n_experts}")
    print(f"  use_act        = {cfg.use_act}")
    print()

    print(f"Parameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:20s}: {count:>12,}")
    print()

    # Forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    print(f"Input shape:       {tuple(input_ids.shape)}")
    print(f"Logits shape:      {tuple(output.logits.shape)}")
    print(f"Aux loss:          {output.aux_loss.item():.6f}")
    print(f"Ponder cost:       {output.ponder_cost.item():.6f}")
    print()

    # Verify output shapes
    assert output.logits.shape == (batch_size, seq_len, cfg.vocab_size), \
        f"Unexpected logits shape: {output.logits.shape}"
    assert output.aux_loss.isfinite(), "Aux loss is not finite!"
    assert output.ponder_cost.isfinite(), "Ponder cost is not finite!"
    print("[OK] All shape and finiteness checks passed!")


def demo_generation(cfg: MythosConfig) -> None:
    """Demonstrate autoregressive generation."""
    print_separator("Autoregressive Generation")

    model = OpenMythos(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    max_new = 16

    print(f"Prompt tokens:     {prompt.squeeze().tolist()}")
    print(f"Generating {max_new} new tokens...")

    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=max_new,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
        )

    new_tokens = generated[0, 8:].tolist()
    print(f"Generated tokens:  {new_tokens}")
    print(f"Total length:      {generated.shape[1]}")
    assert generated.shape == (1, 8 + max_new), \
        f"Unexpected generated shape: {generated.shape}"
    assert all(0 <= t < cfg.vocab_size for t in new_tokens), \
        "Generated tokens out of vocab range!"
    print("[OK] Generation check passed!")


def main() -> None:
    print("=" * 60)
    print("  OpenMythos — Architecture Demo")
    print("  Recurrent-Depth Transformer with Mixture-of-Experts")
    print("=" * 60)

    # ── Small config for demo (fits on CPU) ──────────────────────────────
    base_cfg = dict(
        vocab_size=1000,
        dim=256,
        max_seq_len=512,
        prelude_layers=2,
        coda_layers=2,
        max_loop_iters=4,
        n_heads=8,
        n_kv_heads=2,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_head_dim=32,
        v_head_dim=32,
        qk_rope_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=8,
    )

    # ── Demo 1: MLA Attention ────────────────────────────────────────────
    mla_cfg = MythosConfig(**base_cfg, attn_type="mla")
    demo_forward_pass(mla_cfg, "Multi-Latent Attention (MLA)")

    # ── Demo 2: GQA Attention ────────────────────────────────────────────
    gqa_cfg = MythosConfig(**base_cfg, attn_type="gqa")
    demo_forward_pass(gqa_cfg, "Grouped Query Attention (GQA)")

    # ── Demo 3: MLA with ACT ─────────────────────────────────────────────
    act_cfg = MythosConfig(**base_cfg, attn_type="mla", use_act=True)
    demo_forward_pass(act_cfg, "MLA + Adaptive Computation Time (ACT)")

    # ── Demo 4: Generation ───────────────────────────────────────────────
    demo_generation(mla_cfg)

    print_separator("All demos completed successfully!")


if __name__ == "__main__":
    main()
