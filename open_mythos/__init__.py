"""
OpenMythos — Open-source Recurrent-Depth Transformer with Mixture-of-Experts.

A first-principles theoretical reconstruction of the Claude Mythos architecture,
implemented in PyTorch. Features a three-part Prelude → Recurrent Loop → Coda
pipeline with sparse MoE routing, Multi-Latent Attention, and adaptive computation.
"""

from open_mythos.config import MythosConfig
from open_mythos.model import OpenMythos

__all__ = ["OpenMythos", "MythosConfig"]
__version__ = "0.1.0"
