"""Tests for CharTokenizer and BPETokenizer.

Test cases:
    - CharTokenizer encode/decode roundtrip
    - CharTokenizer vocab_size == 10
    - BPETokenizer vocab_size == 4096 (after training)
    - batch_encode padding + attention_mask shapes
    - get_tokenizer("v3.1") → CharTokenizer, get_tokenizer("v4.0") → BPETokenizer
"""
# TODO: Implement tokenizer tests
