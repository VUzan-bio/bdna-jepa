"""Logging utilities: W&B integration + console formatting.

Functions:
    setup_wandb(project, config)  — Initialize W&B run
    log_metrics(step, metrics)    — Log dict to W&B + console
    log_checkpoint(path, metrics) — Log checkpoint save event

← NEW (was inline in 01_pretrain_jepa.py)
"""
# TODO: Extract W&B logging from 01_pretrain_jepa.py
