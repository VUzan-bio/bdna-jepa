# B-JEPA

**Bacterial Joint-Embedding Predictive Architecture**

Self-supervised foundation model for bacterial genomics, pretrained on 417 prokaryotic species via joint masked language modeling and latent-space prediction.

[Model](https://huggingface.co/orgava/dna-bacteria-jepa) ·
[GUARD Pipeline](https://github.com/VUzan-bio/guard) ·
[Architecture](#architecture) ·
[Quick Start](#quick-start) ·
[Citation](#citation)

---

## Overview

B-JEPA learns general-purpose representations of bacterial DNA through a dual-objective self-supervised framework. A context encoder processes partially-masked genomic sequences, a predictor maps context CLS embeddings to target CLS embeddings in latent space, and an exponential moving average (EMA) target encoder provides the prediction targets. A parallel masked language model (MLM) head grounds the representations at the token level. VICReg regularization prevents representational collapse.

The resulting embeddings capture biological structure — GC content gradients, species-level clustering, and sequence-level functional context — without any supervised labels, and serve as the scoring backbone for [GUARD](https://github.com/VUzan-bio/guard), a CRISPR-Cas12a diagnostic design pipeline targeting multidrug-resistant tuberculosis (MDR-TB).

## Models

| Version | Parameters | Architecture | Seq Length | Tokenizer | RankMe | Weights |
|---------|-----------|-------------|-----------|-----------|--------|---------|
| v3.1 | 8.5M | 6L × 384D × 6H | 512 | Char-level | 372 / 384 | [checkpoint](https://huggingface.co/orgava/dna-bacteria-jepa) |
| v4.0 | 48M | 12L × 576D × 9H | 1024 | BPE (4096) | — | *training* |

## Architecture

<img width="2816" height="1317" alt="Gemini_Generated_Image_3kttjv3kttjv3ktt" src="https://github.com/user-attachments/assets/ce22938e-9695-41d2-ba3b-5f8a9f08a1bd" />

```
L_total = w_mlm · L_MLM + w_jepa · L_JEPA + w_var · L_var + w_cov · L_cov
```

| Loss | Formulation | Role |
|------|-------------|------|
| L_MLM | Cross-entropy on 15% masked tokens | Token-level sequence understanding |
| L_JEPA | MSE between predicted and target CLS | Sequence-level functional context |
| L_var | Hinge: per-dimension std ≥ 1 | Prevents complete collapse |
| L_cov | Off-diagonal covariance penalty | Prevents dimensional collapse |

**Encoder.** Transformer with BPE tokenization (v4.0), RoPE positional embeddings, and bfloat16 mixed precision. The predictor uses a 0.33× width bottleneck following [I-JEPA](https://arxiv.org/abs/2301.08243), forcing the encoder to produce richer representations rather than offloading capacity to the predictor.

**EMA target.** Cosine schedule from τ = 0.996 → 1.0 over training. The target encoder is never trained directly — it tracks the context encoder via exponential moving average, providing stable prediction targets whose complexity grows with encoder quality.

**Fragment JEPA.** Cross-fragment prediction across non-contiguous genomic regions, extending the standard single-sequence JEPA to capture longer-range genomic context.

### Design decisions: v3.1 → v4.0

| | v3.1 | v4.0 | Rationale |
|---|------|------|-----------|
| Scale | 8.5M | 48M | 5.6× capacity for 417-species corpus |
| Tokenizer | Char-level | BPE (4096) | ~5× compression; learns motifs natively |
| Positional encoding | Learned | RoPE | Length generalization beyond training context |
| Predictor width | 384D (1×) | 192D (0.33×) | I-JEPA bottleneck forces richer encoder |
| Collapse prevention | SIGReg | VICReg (var + cov) | SIGReg collapsed at epoch 80 in v3.1 |
| Loss balancing | Static weights | GradNorm (α = 1.5) | JEPA gradient was ~512× larger than MLM |
| JEPA target | Token-block | CLS latent | 2D block masking from I-JEPA fails on 1D DNA |
| Fragment JEPA | — | ✓ | Cross-fragment genome context |

## Quick Start

### Installation

```bash
git clone https://github.com/VUzan-bio/bdna-jepa.git && cd bdna-jepa
pip install -e ".[all]"
```

### Pretraining

```bash
# v4.0 — 48M model (requires A100 or equivalent)
python scripts/pretrain.py --config configs/training/v4.0.yaml

# v3.1 — 8.5M baseline (any GPU)
python scripts/pretrain.py --config configs/training/v3.1.yaml
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/v4.0/best.pt \
    --version v4.0
# → RankMe, kNN, linear probe, GC R², clustering, UMAP
```

### Feature extraction

```python
from bdna_jepa import load_encoder

encoder = load_encoder("orgava/dna-bacteria-jepa", version="v4.0")
embeddings = encoder.encode_sequences(["ATCGATCG..."])  # → (1, 576)
```

## Training

**Optimizer:** AdamW (lr = 1e-3, cosine decay → 1e-6), bfloat16 AMP, batch size 256, 300 epochs.

**Data:** 417 prokaryotic species, fragmented into overlapping 1024-token windows. Species are sampled proportionally to genome size with a smoothing cap to prevent dominance by large genomes.

**Monitoring:** Training health is tracked via RankMe (effective embedding rank), representation variance/covariance, dead neuron fraction, UMAP visualizations colored by GC content and species identity, and downstream linear probes evaluated at epoch boundaries.

See `configs/training/v4.0.yaml` for the full configuration.

## Downstream: GUARD

[**GUARD**](https://github.com/VUzan-bio/guard) (Guided Universal Assay for Resistance Diagnostics) uses B-JEPA embeddings as the activity scoring engine within a 9-module automated pipeline for CRISPR-Cas12a diagnostic design:

```
WHO mutation catalogue
  → target resolution (M1)
  → PAM scanning (M2)
  → candidate filtering (M3)
  → off-target screening (M4)
  → B-JEPA activity scoring (M5)    ← embeddings used here
  → mismatch pair generation (M5.5)
  → synthetic mismatch enhancement (M6)
  → multiplex optimization (M7)
  → RPA primer co-design (M8)
  → panel assembly (M9)
```

B-JEPA embeddings score crRNA guide candidates by encoding spacer-target context and predicting Cas12a cleavage efficiency, replacing hand-crafted feature engineering with learned genomic representations.

**Target application:** 14-plex electrochemical biosensor on laser-induced graphene electrodes for point-of-care MDR-TB detection. Part of a BRIDGE Discovery project at ETH Zürich (deMello Group, D-CHAB) in collaboration with CSEM.

```python
# Using B-JEPA within GUARD
from bdna_jepa import load_encoder

encoder = load_encoder("orgava/dna-bacteria-jepa", version="v4.0")
# → score crRNA candidates, predict mismatch tolerance, rank multiplex panels
```

## Repository Structure

```
bdna-jepa/
├── bdna_jepa/                    # Core library (pip install -e .)
│   ├── models/                   #   Encoder, predictor, JEPA wrapper
│   ├── losses/                   #   JEPA + MLM + VICReg + GradNorm
│   ├── data/                     #   BPE tokenizer, dataset, masking strategies
│   ├── training/                 #   Trainer, EMA, checkpointing
│   ├── evaluation/               #   Linear probing, clustering, visualization
│   ├── utils/                    #   RankMe, feature extraction, logging
│   ├── config.py                 #   Versioned dataclass configs
│   └── hub.py                    #   HuggingFace load/save
├── configs/                      #   YAML configs: model, training, evaluation
├── scripts/                      #   pretrain · evaluate · visualize · export
├── tools/                        #   download_genomes · extract · tokenize
├── tests/
└── docs/
```

## References

- **I-JEPA** — Assran et al. (2023). [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.](https://arxiv.org/abs/2301.08243) *CVPR 2023.* Predictor bottleneck, EMA schedule.
- **JEPA-DNA** — Larey et al. (2026). Dual MLM + CLS-JEPA for DNA sequences.
- **C-JEPA** — Mo et al. (2024). VICReg regularization for JEPA.
- **V-JEPA** — Bardes et al. (2024). [Revisiting Feature Prediction for Learning Visual Representations from Video.](https://arxiv.org/abs/2401.04325) Multi-scale masking.
- **ProkBERT** — Ligeti et al. (2024). [ProkBERT: Genomic Language Models for Microbiome Analysis.](https://doi.org/10.1093/nargab/lqae065) Prokaryote MLM baseline.
- **GradNorm** — Chen et al. (2018). [GradNorm: Gradient Normalization for Adaptive Loss Balancing.](https://arxiv.org/abs/1711.02257) *ICML 2018.* Multi-task gradient balancing.
- **VICReg** — Bardes et al. (2022). [VICReg: Variance-Invariance-Covariance Regularization.](https://arxiv.org/abs/2105.04906) *ICLR 2022.* Variance-covariance regularization.

## Citation

```bibtex
@misc{uzan2026bjepa,
    title   = {{B-JEPA}: Self-Supervised Bacterial Genomic Foundation Model
               via Joint-Embedding Predictive Architectures},
    author  = {Uzan, Valentin},
    year    = {2026},
    url     = {https://github.com/VUzan-bio/bdna-jepa}
}
```

## License

MIT
