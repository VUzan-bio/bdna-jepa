#!/usr/bin/env python3
"""B-JEPA pretraining entry point.

Usage:
    python scripts/pretrain.py --config configs/training/v4.0.yaml
    python scripts/pretrain.py --config configs/training/v3.1.yaml
    python scripts/pretrain.py --config configs/training/v4.0.yaml --resume outputs/checkpoints/v4.0/epoch100.pt

Flow: load config → build tokenizer → build dataset → build model → BJEPATrainer.train()

← scripts/01_pretrain_jepa.py (thin wrapper — loop logic in bdna_jepa/training/trainer.py)
"""
# TODO: Implement argparse + config loading + wire to BJEPATrainer
