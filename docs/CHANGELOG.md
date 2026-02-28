# Changelog

## v4.0 (in progress)
- 12L×576D×9H encoder (48M params, 5.6× v3.1)
- BPE tokenizer (vocab 4096, ~5× compression)
- Rotary positional encoding (RoPE)
- Dual objective: MLM + CLS-level JEPA
- VICReg collapse prevention (replaces SIGReg)
- GradNorm loss balancing (α=1.5)
- Fragment-level JEPA for genome context
- Asymmetric predictor bottleneck (192D, 0.33×)

## v3.1
- 6L×384D×6H encoder (8.5M params)
- Character-level tokenizer
- Learned positional encoding
- SIGReg collapse prevention
- Trained 80 epochs on 7.6M fragments (417 species)
- RankMe: 372/384 (97% utilization)
- kNN@5: 3.0%, linear probe: 6.3%, GC R²: 0.257
- Published to HuggingFace: orgava/dna-bacteria-jepa
