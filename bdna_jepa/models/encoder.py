"""Transformer encoder backbone for B-JEPA.

Supports v3.1 (char-level, learned pos) and v4.0 (BPE, RoPE).
Outputs per-token embeddings and a pooled [CLS] representation.

Classes:
    RotaryEmbedding      — RoPE for length generalization (v4.0)
    MultiHeadAttention   — MHA with optional RoPE
    TransformerBlock     — Pre-norm transformer block
    TransformerEncoder   — Full encoder: token_emb + CLS + layers + final_norm

Key methods:
    forward(tokens, attention_mask, return_all_tokens) → {"cls": (B,D), "tokens": (B,L+1,D)}
    encode(tokens, attention_mask) → (B, D)  # convenience, CLS only

← src/cas12a/encoder.py (rewritten: added RoPE, pre-norm, config-driven architecture)
"""
# TODO: Port encoder from src/cas12a/encoder.py
#   - Add RotaryEmbedding class
#   - Switch to pre-norm (LayerNorm before attention, not after)
#   - Make architecture driven by EncoderConfig dataclass
#   - Support both learned pos (v3.1) and RoPE (v4.0) via config
