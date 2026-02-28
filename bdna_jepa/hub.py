"""HuggingFace Hub integration: load/save/export B-JEPA models.

Functions:
    load_encoder(repo_or_path, version)  — Load pretrained encoder from HF or local
    load_full_model(checkpoint, config)  — Load full BJEPA for continued training
    save_checkpoint(model, optimizer, epoch, metrics, path) — Save checkpoint

← NEW (replaces manual torch.load scattered across old scripts)
"""
# TODO: Implement HF upload/download + checkpoint save/load
