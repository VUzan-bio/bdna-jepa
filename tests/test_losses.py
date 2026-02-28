"""Tests for loss functions.

Test cases:
    - VICReg variance_loss > 0.9 on collapsed (all-ones) input
    - VICReg variance_loss < 0.1 on healthy (std>gamma) input
    - JEPALoss > 0 and has grad_fn
    - GradNorm weights are all positive, balance towards equal grad norms
    - BJEPACriterion total = weighted sum of components
"""
# TODO: Implement loss function tests
