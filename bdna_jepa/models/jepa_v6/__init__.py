"""B-JEPA v6.0 — True JEPA for Bacterial DNA.

A genuine Joint-Embedding Predictive Architecture where JEPA is the PRIMARY
learning signal and MLM serves as an anti-collapse guard.

Key features:
  - Asymmetric masking: context sees 30-50% of tokens (50-70% masked)
  - Cross-attention predictor: target position queries attend to context KV
  - JEPA as primary loss (5.0) with MLM anti-collapse guard (0.5)
  - SIGReg regularization (LeJEPA, provably optimal)
  - Proper encoder: RoPE + RMSNorm + SwiGLU + QK-Norm
  - GC content debiasing via gradient reversal

Usage:
    python bdna_jepa/models/jepa_v6/pretrain_v6.py \\
        --data-path data/processed/pretrain_2M.csv \\
        --tokenizer-path data/tokenizer/bpe_4096.json
"""
