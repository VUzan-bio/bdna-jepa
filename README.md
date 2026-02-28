# B-JEPA: Bacterial Joint-Embedding Predictive Architecture

<p align="center">
  <em>Self-supervised foundation model for bacterial genomics</em>
</p>

<p align="center">
  <a href="https://huggingface.co/orgava/dna-bacteria-jepa">🤗 Model</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="docs/">Docs</a>
</p>

---

## Models

| Version | Params | Dim | Layers | Heads | Seq Len | Tokenizer  | RankMe   | Status |
|---------|--------|-----|--------|-------|---------|------------|----------|--------|
| v3.1    | 8.5M   | 384 | 6      | 6     | 512     | Char-level | 372/384  | ✅ [Released](https://huggingface.co/orgava/dna-bacteria-jepa) |
| v4.0    | 48M    | 576 | 12     | 9     | 1024    | BPE (4096) | TBD      | 🔧 In progress |

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │         B-JEPA Pretraining            │
                    │                                       │
                    │   ┌──────────┐    ┌──────────┐       │
                    │   │ Context  │    │  Target  │       │
                    │   │ Encoder  │    │ Encoder  │ (EMA) │
                    │   └────┬─────┘    └────┬─────┘       │
                    │        │               │              │
                    │   ┌────▼─────┐         │              │
                    │   │Predictor │─── L_JEPA(CLS)         │
                    │   │(narrow)  │         │              │
                    │   └──────────┘         │              │
                    │        │               │              │
                    │   L_MLM(tokens)   L_VICReg(CLS)      │
                    └──────────────────────────────────────┘
```

### Pretraining objectives

| Loss      | Purpose                                         | Balancing |
|-----------|-------------------------------------------------|-----------|
| `L_MLM`   | Token-level masked language modeling             | GradNorm  |
| `L_JEPA`  | CLS-token latent prediction (context → target)  | GradNorm  |
| `L_var`   | VICReg variance — prevent complete collapse      | GradNorm  |
| `L_cov`   | VICReg covariance — prevent dimensional collapse | GradNorm  |

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Pretrain

```bash
python scripts/pretrain.py --config configs/training/v4.0.yaml
python scripts/pretrain.py --config configs/training/v3.1.yaml          # reproduce baseline
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/v4.0/best.pt \
    --config configs/evaluation/full.yaml
```

### Use embeddings

```python
from bdna_jepa import load_encoder

encoder = load_encoder("orgava/dna-bacteria-jepa", version="v4.0")
embeddings = encoder.encode_sequences(["ATCGATCGATCG..."])
# → (1, 576) tensor
```

## v3.1 → v4.0 Changes

| Component            | v3.1                      | v4.0                             |
|----------------------|---------------------------|----------------------------------|
| Encoder              | 6L × 384D × 6H           | 12L × 576D × 9H                 |
| Predictor            | 4L × 384D (same width)   | 4L × 192D (bottleneck)          |
| Tokenizer            | Character-level (ACGTN)   | BPE (vocab 4096)                 |
| Position encoding    | Learned                   | Rotary (RoPE)                    |
| Collapse prevention  | SIGReg                    | VICReg (var + cov)               |
| Loss balancing       | Static weights            | GradNorm (α=1.5)                |
| JEPA target          | Token-block prediction    | CLS-token latent prediction      |
| Fragment JEPA        | ✗                         | ✓ (cross-fragment genome-level)  |
| Sequence length      | 512                       | 1024                             |

## Downstream

- **[SABER](https://github.com/VUzan-bio/saber)** — CRISPR-Cas12a crRNA design pipeline using B-JEPA embeddings for MDR-TB diagnostics

## Citation

```bibtex
@misc{uzan2026bjepa,
  title   = {B-JEPA: Self-Supervised Bacterial Genomic Foundation Model
             via Joint-Embedding Predictive Architectures},
  author  = {Uzan, Valentin},
  year    = {2026},
  url     = {https://github.com/VUzan-bio/bdna-jepa}
}
```

## License

MIT
