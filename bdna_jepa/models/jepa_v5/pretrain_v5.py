#!/usr/bin/env python3
"""B-JEPA v5.0 — MLM + JEPA-CLS Hybrid for Bacterial DNA
==========================================================

Architecture follows JEPA-DNA (Larey et al., arXiv 2602.17162, Feb 2026):
  MLM as primary loss  → per-token cross-entropy prevents encoder collapse
  JEPA on [CLS] only   → global semantic grounding MLM misses
  VICReg on [CLS]      → prevents known CLS collapse (BEPA, JEPA-DNA)
  GC adversary          → debiases GC-content shortcut

Why v5 exists (v4.0–v4.5 post-mortem):
  v4.0–v4.4: CLS→CLS JEPA collapsed (RankMe 32→6) — trivial identity solution.
  v4.5a-g: Per-token JEPA collapsed (RankMe 18→6) — DNA BPE tokens have
  extreme local redundancy; predictor interpolates from neighbors trivially
  regardless of masking ratio (15–70%), strategy (block/random), or
  regularizer (SIGReg/VICReg on pooled/tokens, weights 1–100).

  Root cause: In 1D DNA with BPE, per-token latent prediction is always
  trivially solvable. Images/video don't have this problem because 2D
  patches contain semantically diverse content.

  Solution (from JEPA-DNA, C-JEPA, BEPA literature):
    MLM provides inherent anti-collapse — different positions MUST produce
    different hidden states to predict different tokens.
    JEPA on CLS adds global semantics MLM misses.

Data:   pretrain_2M.csv (2M fragments, 6326 genomes), BPE vocab=4096
Encoder: 12L × 576D × 9H, pre-norm, GELU, [CLS] token, learned pos
Target:  EMA copy (τ = 0.996→1.0)
Predictor: 3L × 384D × 6H, CLS-only (not per-token)
MLM head: LN → Dense → GELU → Linear(vocab_size)
Masking: Span-based, 20–30% (standard MLM range)

References:
  [1] Larey et al. "JEPA-DNA." arXiv:2602.17162, Feb 2026.
  [2] Mo et al. "C-JEPA." NeurIPS 2024 Spotlight.
  [3] Balestriero & LeCun. "LeJEPA." arXiv:2511.08544, Nov 2025.
  [4] BERT-JEPA/BEPA. arXiv:2601.00366, Dec 2025.

Usage:
  cd /workspace/bdna-jepa
  python bdna_jepa/models/pretrain_v5.py \
      --data-path data/processed/pretrain_2M.csv \
      --tokenizer-path data/tokenizer/bpe_4096.json \
      --epochs 20 --batch-size 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        return iterable

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Encoder — Pre-norm Transformer WITH [CLS] token
# ═══════════════════════════════════════════════════════════════════════════════

class DNAEncoder(nn.Module):
    """Transformer encoder with prepended [CLS] token.

    Returns both per-token embeddings (for MLM) and CLS embedding (for JEPA).
    Uses nn.TransformerEncoderLayer with pre-norm (norm_first=True).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 576,
        num_layers: int = 12,
        num_heads: int = 9,
        ff_dim: int = 2304,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # +1 for CLS position at index 0
        self.pos_embedding = nn.Embedding(max_seq_len + 1, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

    def get_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, L) bool mask where True = valid token."""
        return tokens != self.pad_token_id

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L) token IDs
            attention_mask: (B, L) bool, True = valid

        Returns dict:
            cls:    (B, D) CLS embedding
            tokens: (B, L, D) per-token embeddings (excluding CLS position)
            all:    (B, L+1, D) full sequence including CLS at position 0
        """
        B, L = tokens.shape
        if attention_mask is None:
            attention_mask = self.get_attention_mask(tokens)

        # Token embeddings
        x = self.token_embedding(tokens)  # (B, L, D)

        # Prepend CLS
        cls_expanded = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_expanded, x], dim=1)  # (B, L+1, D)

        # Positional embeddings (0=CLS, 1..L=tokens)
        positions = torch.arange(L + 1, device=tokens.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.embed_dropout(x)

        # Attention mask: prepend True for CLS
        cls_mask = torch.ones(B, 1, device=tokens.device, dtype=torch.bool)
        full_mask = torch.cat([cls_mask, attention_mask.bool()], dim=1)  # (B, L+1)
        key_padding_mask = ~full_mask  # TransformerEncoder expects True=ignore

        # Encode
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]           # (B, D) — CLS
        token_out = x[:, 1:, :]        # (B, L, D) — per-token (no CLS)

        return {"cls": cls_out, "tokens": token_out, "all": x}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MLM Head
# ═══════════════════════════════════════════════════════════════════════════════

class MLMHead(nn.Module):
    """Masked Language Model head: LN → Dense → GELU → Linear(vocab)."""

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """(B, L, D) → (B, L, vocab_size)"""
        return self.output(self.act(self.dense(self.norm(token_embeddings))))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. JEPA CLS Predictor — Lightweight bottleneck
# ═══════════════════════════════════════════════════════════════════════════════

class CLSPredictor(nn.Module):
    """3-layer Transformer predictor operating on CLS token only.

    Following JEPA-DNA: predictor_dim=384, 3 layers, 6 heads.
    Input: context CLS (D_enc) → project to D_pred → 3× self-attn blocks
    → project back to D_enc → predicted target CLS.
    """

    def __init__(self, embed_dim: int = 576, predictor_dim: int = 384,
                 depth: int = 3, num_heads: int = 6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(predictor_dim),
                "ff": nn.Sequential(
                    nn.Linear(predictor_dim, predictor_dim * 2),
                    nn.GELU(),
                    nn.Linear(predictor_dim * 2, predictor_dim),
                ),
                "norm2": nn.LayerNorm(predictor_dim),
                "ff2": nn.Sequential(
                    nn.Linear(predictor_dim, predictor_dim * 2),
                    nn.GELU(),
                    nn.Linear(predictor_dim * 2, predictor_dim),
                ),
            }))
        self.final_norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """(B, D_enc) → (B, D_enc) predicted target CLS."""
        x = self.input_proj(cls_embedding)  # (B, D_pred)
        for block in self.blocks:
            x = x + block["ff"](block["norm1"](x))
            x = x + block["ff2"](block["norm2"](x))
        x = self.final_norm(x)
        return self.output_proj(x)  # (B, D_enc)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VICReg on CLS (JEPA-DNA style)
# ═══════════════════════════════════════════════════════════════════════════════

def vicreg_loss(z: torch.Tensor, var_weight: float = 25.0, cov_weight: float = 1.0,
                gamma: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """VICReg variance + covariance on CLS embeddings.

    Args:
        z: (B, D) batch of CLS embeddings
        var_weight: penalty for per-dim std < gamma
        cov_weight: penalty for off-diagonal correlations

    Returns:
        total loss, metrics dict
    """
    z = z.float()
    B, D = z.shape

    # Variance: each dimension should have std >= gamma
    std = z.std(dim=0)
    var_loss = F.relu(gamma - std).mean()

    # Covariance: decorrelate dimensions
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(B - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / D

    total = var_weight * var_loss + cov_weight * cov_loss

    return total, {
        "vicreg_total": total.item(),
        "vicreg_var": var_loss.item(),
        "vicreg_cov": cov_loss.item(),
        "cls_std_mean": std.mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GC Adversary (gradient reversal)
# ═══════════════════════════════════════════════════════════════════════════════

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GCAdversary(nn.Module):
    """Gradient-reversal adversary for GC content debiasing."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        return self.net(_GradReverse.apply(emb, lam)).squeeze(-1)

    @staticmethod
    def ganin_lambda(epoch: int, total: int, gamma: float = 10.0) -> float:
        p = epoch / max(total - 1, 1)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def compute_gc_content(tokens: torch.Tensor, pad_id: int, gc_ids: Set[int]) -> torch.Tensor:
    """Per-sequence GC fraction."""
    valid = tokens != pad_id
    lengths = valid.sum(dim=1).float().clamp(min=1)
    gc_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for tid in gc_ids:
        gc_mask |= (tokens == tid)
    return (gc_mask & valid).sum(dim=1).float() / lengths


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Span Masking (MLM-style, not I-JEPA block masking)
# ═══════════════════════════════════════════════════════════════════════════════

def span_mask(
    tokens: torch.Tensor,
    mask_ratio: float,
    pad_id: int,
    mask_id: int,
    mean_span_len: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate span-masked tokens + mask indicator.

    Following JEPA-DNA: span-based masking at 20–30%.

    Args:
        tokens: (B, L) original token IDs
        mask_ratio: fraction of eligible tokens to mask
        pad_id: padding token ID
        mask_id: mask token ID
        mean_span_len: geometric distribution parameter for span lengths

    Returns:
        masked_tokens: (B, L) with [MASK] at masked positions
        mask_bool: (B, L) True at masked positions
    """
    B, L = tokens.shape
    device = tokens.device
    masked_tokens = tokens.clone()
    mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)

    for b in range(B):
        valid = (tokens[b] != pad_id)
        valid_idx = valid.nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        n_mask = max(1, int(n_valid * mask_ratio))

        # Generate spans with geometric length distribution
        masked_count = 0
        occupied = torch.zeros(n_valid, dtype=torch.bool)

        while masked_count < n_mask:
            # Sample span length from geometric distribution
            span_len = max(1, int(np.random.geometric(1.0 / mean_span_len)))
            span_len = min(span_len, n_mask - masked_count, n_valid)

            # Sample start position
            max_start = n_valid - span_len
            if max_start < 0:
                break

            start = random.randint(0, max_start)

            # Check overlap (allow some, like standard MLM)
            actual_pos = valid_idx[start:start + span_len]
            new_mask = ~occupied[start:start + span_len]
            new_count = new_mask.sum().item()

            if new_count == 0:
                continue

            occupied[start:start + span_len] = True
            mask_bool[b, actual_pos] = True
            masked_tokens[b, actual_pos] = mask_id
            masked_count += new_count

            if masked_count >= n_mask:
                break

    return masked_tokens, mask_bool


# ═══════════════════════════════════════════════════════════════════════════════
# 7. B-JEPA v5 Model — MLM + JEPA-CLS Hybrid
# ═══════════════════════════════════════════════════════════════════════════════

class BJEPAv5(nn.Module):
    """MLM + JEPA-CLS hybrid for bacterial DNA.

    Forward:
      1. Span-mask input → [MASK] tokens
      2. Context encoder: masked input → per-token embeds + CLS
      3. MLM head: per-token embeds → logits (cross-entropy at masked pos)
      4. Target encoder (EMA): full input → target CLS
      5. CLS predictor: context CLS → predicted target CLS (cosine loss)
      6. VICReg on CLS embeddings (variance + covariance)
      7. GC adversary on CLS (gradient reversal)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 576,
        num_layers: int = 12,
        num_heads: int = 9,
        ff_dim: int = 2304,
        max_seq_len: int = 512,
        predictor_dim: int = 384,
        predictor_depth: int = 3,
        predictor_heads: int = 6,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.embed_dim = embed_dim
        self.ema_start = ema_start
        self.ema_end = ema_end
        self._ema_decay = ema_start

        # Context encoder (trainable)
        self.context_encoder = DNAEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_layers=num_layers, num_heads=num_heads,
            ff_dim=ff_dim, max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
        )

        # Target encoder (frozen, EMA-updated)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # MLM head
        self.mlm_head = MLMHead(embed_dim, vocab_size)

        # JEPA CLS predictor (lightweight)
        self.cls_predictor = CLSPredictor(
            embed_dim=embed_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, num_heads=predictor_heads,
        )

        # GC adversary
        self.gc_adversary = GCAdversary(embed_dim, hidden_dim=64)

    def set_ema_decay(self, progress: float) -> float:
        """Cosine EMA schedule."""
        t0, t1 = self.ema_start, self.ema_end
        self._ema_decay = t1 - (t1 - t0) * (1 + math.cos(math.pi * progress)) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self):
        tau = self._ema_decay
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

    def forward(
        self,
        tokens: torch.Tensor,
        masked_tokens: torch.Tensor,
        mask_bool: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L) original token IDs
            masked_tokens: (B, L) with [MASK] at masked positions
            mask_bool: (B, L) True at masked positions
            attention_mask: (B, L) True = valid token

        Returns dict with all loss inputs:
            mlm_logits:    (B, L, vocab) — for cross-entropy at mask_bool
            jepa_pred_cls: (B, D) — predicted target CLS
            jepa_target_cls: (B, D) — actual target CLS (detached)
            context_cls:   (B, D) — context encoder CLS
        """
        if attention_mask is None:
            attention_mask = tokens != self.pad_token_id

        # Context encoder on masked input
        ctx_out = self.context_encoder(masked_tokens, attention_mask)
        context_cls = ctx_out["cls"]           # (B, D)
        context_tokens = ctx_out["tokens"]     # (B, L, D)

        # MLM logits from per-token embeddings
        mlm_logits = self.mlm_head(context_tokens)  # (B, L, vocab)

        # Target encoder on full (unmasked) input — no gradient
        with torch.no_grad():
            tgt_out = self.target_encoder(tokens, attention_mask)
            target_cls = tgt_out["cls"]        # (B, D)

        # JEPA CLS prediction
        jepa_pred_cls = self.cls_predictor(context_cls)  # (B, D)

        return {
            "mlm_logits": mlm_logits,
            "jepa_pred_cls": jepa_pred_cls,
            "jepa_target_cls": target_cls.detach(),
            "context_cls": context_cls,
        }

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor, attention_mask=None, use_target=True):
        """Extract CLS embeddings for eval/downstream."""
        enc = self.target_encoder if use_target else self.context_encoder
        if attention_mask is None:
            attention_mask = tokens != self.pad_token_id
        out = enc(tokens, attention_mask)
        return out["cls"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Loss Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_losses(
    model_out: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    mask_bool: torch.Tensor,
    gc_adv_model: GCAdversary,
    gc_target: torch.Tensor,
    gc_lambda: float,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute all v5 losses.

    Returns total loss and metrics dict.
    """
    metrics = {}

    # ── 1. MLM Loss (PRIMARY — prevents encoder collapse) ──
    # Cross-entropy only at masked positions
    logits = model_out["mlm_logits"]  # (B, L, vocab)
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)         # (B*L, V)
    targets_flat = tokens.reshape(-1)            # (B*L,)
    mask_flat = mask_bool.reshape(-1)            # (B*L,)

    # Compute CE loss only at masked positions
    mlm_loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*L,)
    n_masked = mask_flat.sum().clamp(min=1)
    mlm_loss = (mlm_loss_all * mask_flat.float()).sum() / n_masked

    # MLM accuracy at masked positions
    with torch.no_grad():
        preds = logits_flat.argmax(dim=-1)
        mlm_acc = ((preds == targets_flat) & mask_flat).sum().float() / n_masked

    metrics["mlm_loss"] = mlm_loss.item()
    metrics["mlm_acc"] = mlm_acc.item()
    metrics["n_masked"] = n_masked.item()

    # ── 2. JEPA CLS Loss (AUXILIARY — global semantic grounding) ──
    jepa_loss = 1.0 - F.cosine_similarity(
        model_out["jepa_pred_cls"],
        model_out["jepa_target_cls"],
        dim=-1,
    ).mean()

    with torch.no_grad():
        jepa_cos = F.cosine_similarity(
            model_out["jepa_pred_cls"],
            model_out["jepa_target_cls"],
            dim=-1,
        ).mean().item()

    metrics["jepa_loss"] = jepa_loss.item()
    metrics["jepa_cos_sim"] = jepa_cos

    # ── 3. VICReg on CLS (prevents CLS collapse) ──
    vicreg, vicreg_met = vicreg_loss(
        model_out["context_cls"],
        var_weight=args.vicreg_var_weight,
        cov_weight=args.vicreg_cov_weight,
    )
    metrics.update(vicreg_met)

    # ── 4. GC Adversary (debiases GC content) ──
    gc_adv_loss = torch.tensor(0.0, device=tokens.device)
    if args.gc_adv_weight > 0:
        gc_pred = gc_adv_model(model_out["context_cls"], lam=gc_lambda)
        gc_adv_loss = F.mse_loss(gc_pred, gc_target)
    metrics["gc_adv_loss"] = gc_adv_loss.item()

    # ── Total ──
    total = (
        args.mlm_weight * mlm_loss
        + args.jepa_weight * jepa_loss
        + args.vicreg_weight * vicreg
        + args.gc_adv_weight * gc_adv_loss
    )
    metrics["total_loss"] = total.item()

    return total, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RankMe + GC Correlation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rankme(embeddings: torch.Tensor) -> float:
    X = embeddings.float()
    X = X - X.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X)
    except Exception:
        _, s, _ = torch.svd(X)
    s = s[s > 1e-7]
    p = s / s.sum()
    return math.exp(-(p * p.log()).sum().item())


def gc_correlation(tokens, embeddings, pad_id, gc_ids):
    gc = compute_gc_content(tokens, pad_id, gc_ids).cpu().numpy()
    emb = embeddings.float().cpu().numpy()
    try:
        from numpy.linalg import svd
        _, _, Vt = svd(emb - emb.mean(axis=0), full_matrices=False)
        pc1 = emb @ Vt[0]
        r = float(np.corrcoef(gc, pc1)[0, 1])
        return abs(r)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. UMAP + t-SNE Visualization
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_viz(emb_np, gc_np, save_path, epoch):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import umap
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Viz skipped — install umap-learn, sklearn, matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    try:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric='cosine', random_state=42)
        coords = reducer.fit_transform(emb_np)
        sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=gc_np,
                             cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=axes[0], label='GC content')
        axes[0].set_title(f'UMAP — Epoch {epoch} (GC%)')
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Failed: {e}', transform=axes[0].transAxes, ha='center')

    try:
        n = min(3000, len(emb_np))
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        coords = tsne.fit_transform(emb_np[:n])
        sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=gc_np[:n],
                             cmap='RdYlBu_r', s=3, alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label='GC content')
        axes[1].set_title(f't-SNE — Epoch {epoch} (GC%)')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Failed: {e}', transform=axes[1].transAxes, ha='center')

    for ax in axes:
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved viz: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class BPEPretrainDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer_path: str, max_len: int = 512):
        import pandas as pd
        import json
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

        with open(tokenizer_path) as f:
            tok_data = json.load(f)
        self.vocab = tok_data.get('model', tok_data).get('vocab', tok_data.get('vocab', {}))
        self.pad_id = 0
        self.mask_id = self.vocab.get("[MASK]", 1)

        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self._encode = lambda seq: self.tokenizer.encode(seq).ids[:self.max_len]
        except ImportError:
            self._encode = lambda seq: [self.vocab.get(c, 1) for c in seq.upper()][:self.max_len]

        # GC token IDs
        self.gc_token_ids = set()
        for key, val in self.vocab.items():
            k = key.upper().replace('Ġ', '').replace('▁', '')
            if k and all(c in 'GC' for c in k):
                self.gc_token_ids.add(val)
        if not self.gc_token_ids:
            self.gc_token_ids = {2, 3}

        seq_col = 'sequence' if 'sequence' in self.df.columns else self.df.columns[0]
        self.sequences = self.df[seq_col].values
        print(f"  Dataset: {len(self):,} sequences")
        print(f"  GC token IDs: {len(self.gc_token_ids)} tokens")
        print(f"  Mask token ID: {self.mask_id}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self._encode(str(self.sequences[idx]))
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, step, total, warmup, peak, floor=1e-6):
    if step < warmup:
        lr = peak * step / max(warmup, 1)
    else:
        p = (step - warmup) / max(total - warmup, 1)
        lr = floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def mask_ratio_schedule(epoch, total, start=0.20, end=0.30):
    """Gentle curriculum from 20% to 30% masking."""
    p = epoch / max(total - 1, 1)
    t = 0.5 * (1 - math.cos(math.pi * p))
    return start + t * (end - start)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, dl, device, gc_ids, pad_id, max_batches=30):
    model.eval()
    all_emb, all_tok = [], []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        tokens = batch.to(device)
        emb = model.encode(tokens)
        all_emb.append(emb.cpu())
        all_tok.append(tokens.cpu())

    embs = torch.cat(all_emb, 0)
    toks = torch.cat(all_tok, 0)
    rankme = compute_rankme(embs)
    gc_r = gc_correlation(toks, embs, pad_id, gc_ids)
    std = embs.std(dim=0).mean().item()
    norm = embs.norm(dim=1).mean().item()

    return {
        "rankme": rankme, "gc_abs_r": gc_r,
        "std": std, "norm": norm,
        "embeddings": embs, "tokens": toks,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═' * 70}")
    print(f"  B-JEPA v5.0 — MLM + JEPA-CLS Hybrid")
    print(f"{'═' * 70}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Dataset
    dataset = BPEPretrainDataset(args.data_path, args.tokenizer_path, args.max_seq_len)
    pad_id = dataset.pad_id
    mask_id = dataset.mask_id
    gc_ids = dataset.gc_token_ids
    vocab_size = len(dataset.vocab)

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                    prefetch_factor=2)

    # Model
    model = BJEPAv5(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_seq_len=args.max_seq_len,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        pad_token_id=pad_id,
        mask_token_id=mask_id,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
    ).to(device)

    n_enc = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred = sum(p.numel() for p in model.cls_predictor.parameters())
    n_mlm = sum(p.numel() for p in model.mlm_head.parameters())
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Encoder: {n_enc/1e6:.2f}M  CLS Predictor: {n_pred/1e6:.2f}M  MLM Head: {n_mlm/1e6:.2f}M")
    print(f"  Total trainable: {n_total/1e6:.2f}M")
    print(f"  Architecture: {args.num_layers}L × {args.embed_dim}D × {args.num_heads}H + [CLS]")
    print(f"  Predictor: {args.predictor_depth}L × {args.predictor_dim}D (CLS-only)")

    # Optimizer
    opt = torch.optim.AdamW([
        {"params": model.context_encoder.parameters()},
        {"params": model.mlm_head.parameters()},
        {"params": model.cls_predictor.parameters()},
        {"params": model.gc_adversary.parameters(), "lr": args.lr * 5},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    ckpt_dir = Path(args.output_dir) / "checkpoints" / "v5.0"
    viz_dir = ckpt_dir / "viz"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Training: {args.epochs} epochs, warmup={args.warmup_epochs}")
    print(f"  LR: {args.lr:.1e} → {args.min_lr:.1e} (cosine)")
    print(f"  Masking: {args.mask_ratio_start:.0%} → {args.mask_ratio_end:.0%} (span)")
    print(f"  Loss weights: MLM={args.mlm_weight} JEPA={args.jepa_weight} "
          f"VICReg={args.vicreg_weight} GC_adv={args.gc_adv_weight}")
    print(f"  VICReg: var={args.vicreg_var_weight} cov={args.vicreg_cov_weight}")
    print(f"  Steps/epoch: {steps_per_epoch}  Total: {total_steps}")
    print(f"{'═' * 70}\n")

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"bjepa-v5.0-{args.num_layers}L{args.embed_dim}D-mlm-jepa",
                   config=vars(args))

    global_step = 0
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_t = time.time()
        model.train()
        progress = epoch / max(args.epochs - 1, 1)
        ema_tau = model.set_ema_decay(progress)
        mr = mask_ratio_schedule(epoch, args.epochs, args.mask_ratio_start, args.mask_ratio_end)
        gc_lam = GCAdversary.ganin_lambda(epoch, args.epochs) if args.gc_adv_weight > 0 else 0.0

        epoch_metrics = {}
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True, ncols=150)

        for step, tokens in enumerate(pbar, start=1):
            global_step += 1
            lr = cosine_lr(opt, global_step, total_steps, warmup_steps, args.lr, args.min_lr)
            tokens = tokens.to(device, non_blocking=True)
            attn = model.context_encoder.get_attention_mask(tokens)

            # Span masking
            masked_tokens, mask_bool = span_mask(tokens, mr, pad_id, mask_id,
                                                 mean_span_len=args.mean_span_len)

            ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                out = model(tokens, masked_tokens, mask_bool, attn)

                gc_target = compute_gc_content(tokens, pad_id, gc_ids)
                total, met = compute_losses(
                    out, tokens, mask_bool, model.gc_adversary, gc_target, gc_lam, args
                )

            if scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()

            if step % args.grad_accum == 0 or step == steps_per_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                model.update_ema()

            # Accumulate metrics
            for k, v in met.items():
                epoch_metrics.setdefault(k, []).append(v)

            if step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{met['total_loss']:.4f}",
                    "mlm": f"{met['mlm_loss']:.3f}",
                    "acc": f"{met['mlm_acc']:.3f}",
                    "jepa": f"{met['jepa_loss']:.3f}",
                    "jcos": f"{met['jepa_cos_sim']:.3f}",
                    "vr": f"{met['vicreg_var']:.4f}",
                    "mr": f"{mr:.2f}",
                    "lr": f"{lr:.1e}",
                })

                if use_wandb:
                    wandb.log({f"train/{k}": v for k, v in met.items()} |
                              {"schedule/lr": lr, "schedule/mask_ratio": mr},
                              step=global_step)

        # ── Epoch eval ──
        ep_time = time.time() - epoch_t
        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        eval_met = evaluate(model, dl, device, gc_ids, pad_id)

        elapsed = time.time() - t_start
        eta = (elapsed / (epoch + 1)) * (args.epochs - epoch - 1)

        print(f"\n  Epoch {epoch+1}/{args.epochs} ({ep_time:.0f}s, ETA {int(eta//3600)}:{int(eta%3600//60):02d})")
        print(f"  MLM: loss={avg['mlm_loss']:.4f}  acc={avg['mlm_acc']:.3f}")
        print(f"  JEPA: loss={avg['jepa_loss']:.4f}  cos_sim={avg['jepa_cos_sim']:.3f}")
        print(f"  VICReg: var={avg['vicreg_var']:.4f}  cov={avg['vicreg_cov']:.4f}  "
              f"cls_std={avg['cls_std_mean']:.3f}")
        print(f"  GC_adv: {avg['gc_adv_loss']:.5f}")
        print(f"  RankMe: {eval_met['rankme']:.1f}/{args.embed_dim}  "
              f"GC|r|={eval_met['gc_abs_r']:.3f}  std={eval_met['std']:.3f}  "
              f"norm={eval_met['norm']:.1f}")
        print(f"  Mask: {mr:.3f}  LR={lr:.2e}  EMA τ={ema_tau:.4f}")

        if eval_met['rankme'] < 50:
            print(f"  ⚠ WARNING: RankMe={eval_met['rankme']:.1f} < 50!")
        elif eval_met['rankme'] < 200:
            print(f"  ⚡ WATCH: RankMe={eval_met['rankme']:.1f}")
        else:
            print(f"  ✅ HEALTHY: RankMe={eval_met['rankme']:.1f}")

        # Viz
        try:
            emb_np = eval_met['embeddings'].float().numpy()
            gc_np = compute_gc_content(eval_met['tokens'], pad_id, gc_ids).numpy()
            n_viz = min(3000, len(emb_np))
            idx = np.random.choice(len(emb_np), n_viz, replace=False)
            viz_path = str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")
            generate_viz(emb_np[idx], gc_np[idx], viz_path, epoch + 1)
        except Exception as e:
            print(f"  Viz failed: {e}")

        if use_wandb:
            log = {f"epoch/{k}": v for k, v in avg.items()}
            log.update({
                "health/rankme": eval_met["rankme"],
                "health/gc_abs_r": eval_met["gc_abs_r"],
                "health/std": eval_met["std"],
                "health/norm": eval_met["norm"],
                "schedule/ema_tau": ema_tau,
                "epoch": epoch + 1,
            })
            vp = str(viz_dir / f"embeddings_epoch{epoch+1:03d}.png")
            if os.path.exists(vp):
                log["viz/embeddings"] = wandb.Image(vp)
            wandb.log(log, step=global_step)

        if avg['total_loss'] < best_loss:
            best_loss = avg['total_loss']

        # Checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            path = ckpt_dir / f"epoch{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch + 1, "global_step": global_step,
                "model_state_dict": {k: v for k, v in model.state_dict().items()
                                     if not k.startswith("target_encoder.")},
                "target_encoder_state_dict": model.target_encoder.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "metrics": {**avg, **eval_met},
                "config": vars(args),
                "version": "v5.0-mlm-jepa",
            }, path)
            print(f"  Saved: {path}")

    total_time = time.time() - t_start
    h, m = int(total_time // 3600), int(total_time % 3600 // 60)
    print(f"\n{'═' * 70}")
    print(f"  B-JEPA v5.0 complete in {h}h {m}m")
    print(f"  Best loss: {best_loss:.4f}  Final RankMe: {eval_met['rankme']:.1f}")
    print(f"{'═' * 70}")
    if use_wandb:
        wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 15. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description="B-JEPA v5.0 — MLM + JEPA-CLS")
    g = p.add_argument_group("Data")
    g.add_argument("--data-path", default="data/processed/pretrain_2M.csv")
    g.add_argument("--tokenizer-path", default="data/tokenizer/bpe_4096.json")
    g.add_argument("--output-dir", default="outputs")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--warmup-epochs", type=int, default=3)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--grad-accum", type=int, default=1)

    g = p.add_argument_group("Encoder")
    g.add_argument("--embed-dim", type=int, default=576)
    g.add_argument("--num-layers", type=int, default=12)
    g.add_argument("--num-heads", type=int, default=9)
    g.add_argument("--ff-dim", type=int, default=2304)
    g.add_argument("--max-seq-len", type=int, default=512)

    g = p.add_argument_group("JEPA Predictor")
    g.add_argument("--predictor-dim", type=int, default=384)
    g.add_argument("--predictor-depth", type=int, default=3)
    g.add_argument("--predictor-heads", type=int, default=6)
    g.add_argument("--ema-start", type=float, default=0.996)
    g.add_argument("--ema-end", type=float, default=1.0)

    g = p.add_argument_group("Masking")
    g.add_argument("--mask-ratio-start", type=float, default=0.20)
    g.add_argument("--mask-ratio-end", type=float, default=0.30)
    g.add_argument("--mean-span-len", type=float, default=3.0)

    g = p.add_argument_group("Loss Weights")
    g.add_argument("--mlm-weight", type=float, default=1.0)
    g.add_argument("--jepa-weight", type=float, default=1.0)
    g.add_argument("--vicreg-weight", type=float, default=1.0)
    g.add_argument("--vicreg-var-weight", type=float, default=25.0)
    g.add_argument("--vicreg-cov-weight", type=float, default=1.0)
    g.add_argument("--gc-adv-weight", type=float, default=1.0)

    g = p.add_argument_group("Logging")
    g.add_argument("--save-every", type=int, default=5)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--wandb-project", default="bdna-jepa")
    g.add_argument("--wandb-run-name", default=None)

    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
