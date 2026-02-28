"""Tests for TransformerEncoder.

Test cases:
    - v3.1 and v4.0 forward pass shapes: cls=(B,D), tokens=(B,L+1,D)
    - Attention mask handling (padding)
    - v4.0 param count ≈ 48M
    - RoPE vs learned positional encoding
"""
# TODO: Implement encoder shape/param tests
