"""JEPA predictors: narrow bottleneck for latent-space prediction.

Classes:
    Predictor          — Asymmetric CLS→CLS predictor
                         project_in(D→D_pred) → transformer layers → project_out(D_pred→D)
                         Bottleneck ratio 0.33× forces encoder to build richer representations.

    FragmentPredictor  — Cross-fragment genome-level predictor (v4.0+)
                         Given K fragment CLS embeddings from same genome,
                         predicts held-out fragment via cross-attention.

← NEW (was implicit in jepa_pretrain.py, now explicit module + fragment predictor added)
"""
# TODO: Implement Predictor with configurable bottleneck
#   - project_in: Linear(encoder_dim → predictor_dim)
#   - N transformer layers at predictor_dim
#   - project_out: LayerNorm → Linear(predictor_dim → encoder_dim)
# TODO: Implement FragmentPredictor with cross-attention
