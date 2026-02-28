# B-JEPA Architecture Design Decisions

## Design lineage

B-JEPA builds on these key papers:
- **I-JEPA** (Assran et al., CVPR 2023) — Predictor bottleneck principle (0.3× width)
- **C-JEPA** (Mo et al., 2024) — VICReg regularization for JEPA (+4.6% linear probe)
- **V-JEPA** (Bardes et al., 2024) — L1 loss for sequential data, multi-scale masking
- **JEPA-DNA** (Larey et al., Feb 2026) — Dual MLM + CLS-JEPA validated on DNA
- **ProkBERT** (Ligeti et al., NAR 2024) — Prokaryote-specific pretraining baseline

## Key decisions

### 1. Dual objective: MLM + CLS-JEPA
JEPA-DNA showed 2D block masking doesn't transfer to 1D DNA.
Solution: standard 15% MLM for token-level + CLS-to-CLS JEPA for global context.

### 2. VICReg over SIGReg
v3.1 used SIGReg → collapsed at epoch 80 (JEPA gradients ~512× larger than regularizer).
VICReg variance hinge + covariance decorrelation is more robust at small batch sizes.

### 3. GradNorm balancing
v3.1 static weights couldn't balance 4 loss terms. GradNorm (α=1.5) auto-balances.

### 4. BPE tokenization
Character-level wastes capacity on individual nucleotides.
BPE (vocab=4096) gives ~5× compression, learns biologically relevant subwords.

### 5. Rotary positional encoding
Learned positional embeddings cap at max_seq_len.
RoPE generalizes to unseen lengths and encodes relative distances naturally.

### 6. Predictor bottleneck
v3.1 used same-width predictor (384D). I-JEPA showed narrow predictor (0.3×)
forces encoder to do more work → better representations.

## Collapse prevention math

### VICReg
```
L_var = (1/d) Σⱼ max(0, γ − √(Var(zⱼ) + ε))
L_cov = (1/d) Σᵢ≠ⱼ [C(Z)]²ᵢ,ⱼ
```

### RankMe monitoring
```
RankMe(Z) = exp(−Σₖ pₖ log pₖ),  pₖ = σₖ / ‖σ‖₁
```
Collapse signals: RankMe < 50, per-dim std < 0.1, cosine similarity > 0.9
