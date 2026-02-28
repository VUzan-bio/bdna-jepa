"""Model configuration dataclasses for all B-JEPA versions.

Defines:
    EncoderConfig   — Transformer encoder architecture params
    PredictorConfig — JEPA predictor bottleneck params
    JEPAConfig      — Full model config (encoder + predictor + loss weights + EMA)
    TrainingConfig  — Optimizer, schedule, batch, checkpointing
    EvalConfig      — Evaluation pipeline params

Version presets:
    V31_CONFIG — 6L×384D, char-level, learned pos, SIGReg
    V40_CONFIG — 12L×576D, BPE, RoPE, VICReg+GradNorm

YAML I/O:
    load_config() — Load from YAML, resolve version presets

← NEW (consolidates hyperparams scattered across argparse in old scripts)
"""
# TODO: Implement dataclasses + version presets + YAML loader
