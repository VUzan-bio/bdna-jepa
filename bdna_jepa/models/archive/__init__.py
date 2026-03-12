"""Archived model versions (v4.1–v4.5).

These are standalone pretrain scripts from earlier iterations that collapsed
or were superseded. Kept for reference and reproducibility.

Files:
    pretrain_ijepa_v44.py  — v4.4: I-JEPA with cosine loss + multi-block masking
    pretrain_v45.py        — v4.5: Per-token JEPA + SIGReg + curriculum masking
    deploy_v45.sh          — v4.5 deployment script
    setup_v42_step{1,2,3}.sh — v4.2 Vast.ai setup scripts
    patch_v41.sh           — v4.1 hotfix patch

Post-mortem:
    v4.4: CLS->CLS JEPA collapsed (RankMe 32->6) — trivial identity solution.
    v4.5: Per-token JEPA collapsed (RankMe 18->6) — DNA BPE local redundancy
          allows trivial neighbor interpolation regardless of masking strategy.

See jepa_v6/ for the current architecture that resolves these issues.
"""
