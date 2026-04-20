# OpenMythos

**An open-source, first-principles theoretical reconstruction of the Claude Mythos architecture, implemented in PyTorch.**

OpenMythos instantiates a **Recurrent-Depth Transformer (RDT)** with a **Mixture-of-Experts (MoE)** routing mechanism, enabling iterative depth via weight sharing. The architecture follows a three-part structure:

```
Token Embedding → Prelude → Recurrent Block (×K loops) → Coda → LM Head
```

## Architecture

### Three-Part Pipeline

| Stage | Description |
|-------|-------------|
| **Prelude** | Standard transformer blocks (run once) to encode input into initial hidden state `z₀` |
| **Recurrent Block** | Core looped computation with shared weights, iterated `K` times with `z₀` re-injected each step |
| **Coda** | Standard transformer blocks (run once) to produce final representations |

### Recurrent Block Internals

Each loop iteration applies (with shared weights):

1. **Loop-Index Sinusoidal Embedding** — encodes the iteration step as a positional signal
2. **Attention** — Multi-Latent Attention (MLA) or Grouped Query Attention (GQA)
3. **Sparse MoE FFN** — DeepSeekMoE-inspired with routed + shared experts and load-balancing loss
4. **LTI-Stable Recurrent Injection** — `h_{t+1} = α·h_t + (1-α)·(update + z₀)` with learned decay `α ∈ (0,1)`
5. **Depth-Wise LoRA Adapters** — per-iteration low-rank adaptation for differentiated behavior
6. **(Optional) Adaptive Computation Time** — learned halting to exit early per token

### Attention Variants

- **Multi-Latent Attention (MLA)** — DeepSeek-V2 style with KV latent compression and decoupled RoPE. Reduces KV cache from `O(n_heads × head_dim)` to `O(kv_lora_rank)`.
- **Grouped Query Attention (GQA)** — Standard multi-head attention with grouped KV heads.

## Installation

```bash
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- einops ≥ 0.7

## Quick Start

```python
import torch
from open_mythos import OpenMythos, MythosConfig

# Create a small model
cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    max_seq_len=4096,
    prelude_layers=4,
    coda_layers=4,
    max_loop_iters=8,
    attn_type="mla",       # or "gqa"
    n_experts=8,
    n_experts_per_tok=2,
)

model = OpenMythos(cfg)

# Forward pass
input_ids = torch.randint(0, cfg.vocab_size, (1, 128))
output = model(input_ids)

print(output.logits.shape)     # (1, 128, 32000)
print(output.aux_loss.item())  # MoE load-balancing loss
print(output.ponder_cost.item())  # ACT ponder cost (0 if disabled)

# Generation
generated = model.generate(input_ids[:, :16], max_new_tokens=64)
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 32000 | Vocabulary size |
| `dim` | 2048 | Model hidden dimension |
| `max_seq_len` | 4096 | Maximum sequence length |
| `prelude_layers` | 4 | Prelude transformer layers |
| `coda_layers` | 4 | Coda transformer layers |
| `max_loop_iters` | 8 | Max recurrent loop iterations |
| `attn_type` | `"mla"` | `"mla"` or `"gqa"` |
| `n_heads` | 16 | Attention heads |
| `n_kv_heads` | 4 | KV heads (GQA) |
| `kv_lora_rank` | 64 | KV compression rank (MLA) |
| `q_lora_rank` | 128 | Query compression rank (MLA) |
| `qk_head_dim` | 64 | Per-head QK dim (MLA) |
| `v_head_dim` | 64 | Per-head value dim (MLA) |
| `qk_rope_dim` | 32 | RoPE-applied portion of QK dim |
| `n_experts` | 8 | Routed MoE experts |
| `n_shared_experts` | 2 | Always-active shared experts |
| `n_experts_per_tok` | 2 | Top-k experts per token |
| `expert_dim` | 1024 | Expert FFN inner dim |
| `lora_rank` | 16 | Depth-wise LoRA rank |
| `use_act` | False | Enable Adaptive Computation Time |
| `act_threshold` | 0.99 | ACT halting threshold |
| `dropout` | 0.0 | Dropout rate |
| `norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 10000.0 | RoPE base frequency |
| `moe_aux_loss_weight` | 0.01 | Weight for MoE load-balancing loss |

## Project Structure

```
OpenMythos/
├── open_mythos/
│   ├── __init__.py        # Public API
│   ├── config.py          # MythosConfig dataclass
│   ├── layers.py          # RMSNorm, SwiGLU, RoPE, TransformerBlock
│   ├── attention.py       # MLA and GQA attention
│   ├── moe.py             # Sparse MoE FFN with routing
│   ├── recurrent.py       # LTI gate, LoRA adapters, ACT
│   └── model.py           # OpenMythos main model
├── tests/
│   └── test_model.py      # Comprehensive tests
├── example.py             # Demo script
├── pyproject.toml         # Package metadata
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Running the Demo

```bash
python example.py
```

## Training Loss

When training, combine the standard language modeling loss with the auxiliary losses:

```python
output = model(input_ids)
logits = output.logits

# Standard cross-entropy
ce_loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.vocab_size),
    input_ids[:, 1:].reshape(-1),
)

# Total loss = LM loss + MoE load-balancing + ACT ponder cost
total_loss = ce_loss + output.aux_loss + 0.01 * output.ponder_cost
total_loss.backward()
```

## License

MIT
