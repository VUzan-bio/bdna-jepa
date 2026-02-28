"""Loss functions for B-JEPA training.

Classes:
    JEPALoss         — L2 MSE between predicted and target CLS embeddings
    MLMLoss          — Cross-entropy on masked tokens (ignore_index=-100)
    VICRegLoss       — Variance + Covariance regularization (Bardes et al., ICLR 2022)
                       variance_loss: hinge on per-dim std ≥ gamma
                       covariance_loss: squared off-diagonal penalty
    GradNormBalancer — Adaptive gradient magnitude balancing (Chen et al., ICML 2018)
                       Equalizes gradient norms across loss terms via EMA tracking
    BJEPACriterion   — Combined loss:
                       L_total = w_mlm·L_MLM + w_jepa·L_JEPA + w_var·L_var + w_cov·L_cov
                       Weights set by GradNorm or static config

← src/cas12a/jepa_pretrain.py (loss computation part)
← src/cas12a/sparse_loss.py (merged — SIGReg replaced by VICReg)
"""
# TODO: Port loss functions from jepa_pretrain.py
#   - Replace SIGReg with VICReg (variance + covariance)
#   - Add GradNorm balancer with configurable alpha
#   - BJEPACriterion wraps all 4 losses with optional GradNorm
