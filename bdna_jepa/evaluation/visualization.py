"""Embedding visualization: UMAP, t-SNE, spectral plots.

Functions:
    plot_umap(embeddings, labels, **kwargs)              — UMAP colored by species/phylum/GC
    plot_tsne(embeddings, labels, **kwargs)               — t-SNE visualization
    plot_spectral_analysis(embeddings, output_path)       — Singular value spectrum + power law
    plot_training_dynamics(log_path, output_path)          — Loss curves, RankMe over epochs
    generate_all_figures(embeddings, labels, output_dir)   — Full figure suite

← scripts/03_visualize_embeddings.py
← scripts/evaluate_representations.py (figure generation part)
"""
# TODO: Port figure generation from 03_visualize_embeddings.py
# TODO: Consolidate duplicate plotting from evaluate_representations.py
