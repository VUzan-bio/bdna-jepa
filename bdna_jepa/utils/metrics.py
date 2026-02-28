"""Representation quality metrics.

Functions:
    compute_rankme(embeddings)            — Effective rank via Shannon entropy of singular values
                                            RankMe ≈ 1 → collapsed; RankMe → d → full rank
    compute_feature_std(embeddings)       — Mean per-feature standard deviation
    compute_spectral_analysis(embeddings) — Full SVD: singular values, cumvar, power-law alpha

References:
    RankMe: Garrido et al., ICML 2023

← NEW (was inline in evaluate_representations.py)
"""
# TODO: Extract RankMe and spectral analysis from evaluate_representations.py
