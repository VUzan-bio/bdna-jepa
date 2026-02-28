"""Masking strategies for masked language modeling.

Functions:
    random_mask(tokens, mask_ratio=0.15) → (masked_tokens, mask)
        Standard BERT masking: 80% [MASK], 10% random, 10% keep

    block_mask(tokens, mask_ratio=0.15, block_size=5) → (masked_tokens, mask)
        Contiguous span masking — better for DNA local context

← NEW (was inline in training loop of 01_pretrain_jepa.py)
"""
# TODO: Extract masking logic from 01_pretrain_jepa.py into standalone functions
