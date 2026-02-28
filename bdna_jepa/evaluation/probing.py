"""Downstream probing: kNN, linear probe, GC regression.

Functions:
    knn_species_accuracy(embeddings, labels, k_values, cv)
        → {k: {"mean": float, "std": float}} for each k

    linear_probe_classification(embeddings, labels, cv)
        → {"accuracy": float, "std": float}

    gc_regression(embeddings, gc_values, cv)
        → {"r2": float, "r2_std": float}

← scripts/evaluate_representations.py (probing part)
← scripts/04_multilevel_eval.py (probing part)
"""
# TODO: Port kNN/linear probe/GC regression from evaluate_representations.py
