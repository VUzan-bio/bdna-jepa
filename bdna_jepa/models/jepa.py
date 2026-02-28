"""Full B-JEPA model: encoder + EMA target + predictor + MLM head.

Class BJEPA:
    Components:
        context_encoder    — processes masked input (trainable)
        target_encoder     — EMA copy, processes full input (frozen)
        predictor          — narrow bottleneck CLS→CLS prediction
        mlm_head           — LayerNorm → Linear → GELU → Linear(→vocab)
        fragment_predictor — optional genome-level cross-fragment JEPA (v4.0+)

    Key methods:
        forward(tokens, mask_indices, attention_mask)
            → {"mlm_logits", "jepa_pred", "jepa_target", "context_cls", "target_cls"}
        encode(tokens, use_target=True) → (B, D)
        update_target_encoder(decay)
        get_ema_decay(step, total_steps) → float  # cosine schedule 0.996→1.0

← src/cas12a/jepa_pretrain.py (model part only — losses moved to losses/criterion.py)
"""
# TODO: Port model from jepa_pretrain.py
#   - Split model definition from loss computation
#   - Add EMA cosine schedule (was linear in v3.1)
#   - Add fragment_predictor gated by config.use_fragment_jepa
