"""Datasets for B-JEPA pretraining.

Classes:
    BacterialGenomeDataset   — Standard fragment dataset from CSV
                               Columns: sequence, genome (optional), species (optional), gc_content (optional)
                               Returns: {"tokens": Tensor, "idx": int, "gc_content": float}

    GenomeAwareBatchSampler  — Samples K fragments per genome per batch (v4.0 fragment JEPA)
                               Required for cross-fragment prediction objective
                               Builds genome→indices mapping, filters genomes with ≥K fragments

← src/cas12a/dataset.py (extended with genome-aware sampling for fragment JEPA)
"""
# TODO: Port dataset from src/cas12a/dataset.py
# TODO: Add GenomeAwareBatchSampler for fragment JEPA training
