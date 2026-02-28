"""DNA tokenizers: character-level (v3.1) and BPE (v4.0).

Classes:
    CharTokenizer — Character-level A/C/G/T/N + special tokens [PAD/MASK/CLS/SEP/UNK]
                    vocab_size = 10, encode/decode/batch_encode
    BPETokenizer  — Wraps HuggingFace tokenizers library
                    vocab_size = 4096, trained on bacterial corpus
                    ~5× compression vs char-level

Factory:
    get_tokenizer(version="v4.0", tokenizer_path=None) → CharTokenizer | BPETokenizer

← src/cas12a/tokenizer.py (extended with BPE support for v4.0)
"""
# TODO: Port CharTokenizer from src/cas12a/tokenizer.py
# TODO: Add BPETokenizer wrapper around HuggingFace tokenizers
# TODO: get_tokenizer() factory that picks based on version string
