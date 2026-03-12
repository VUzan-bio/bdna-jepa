"""B-JEPA v5.0 — MLM + JEPA-CLS Hybrid.

Superseded by v6.0 (jepa_v6/). Kept for reference.

v5.0 was structurally an MLM model with JEPA decorations:
  - CLS predictor was an MLP denoiser on a single vector (not a true JEPA predictor)
  - 20-30% masking created no information gap
  - JEPA gradients never reached per-token representations

See jepa_v6/ for the true JEPA implementation.
"""
