#!/usr/bin/env python3
"""Extract pretraining sequences: FASTA → windowed fragment CSV.

Usage:
    python tools/extract_sequences.py \
        --input data/genomes/ \
        --output data/processed/pretrain_sequences.csv \
        --window 2048 --stride 512

← scripts/extract_pretraining_sequences.py
"""
# TODO: Port from scripts/extract_pretraining_sequences.py
