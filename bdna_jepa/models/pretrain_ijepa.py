#!/usr/bin/env python3
"""B-JEPA v4.3 — I-JEPA architecture for bacterial DNA.

Merges the working v3 Cas12aJEPA architecture (per-token prediction,
multi-block masking, learnable mask tokens) with v4's BPE tokenizer
and scaled data pipeline.

Architecture verified against:
  - I-JEPA (Assran et al., CVPR 2023): multi-block masking + predictor with mask tokens
  - V-JEPA (Bardes et al., 2024): 90% masking, per-token prediction
  - JEPA-DNA (Larey et al., 2026): MLM + JEPA dual objective, BPE tokenizer
  - GeneJEPA (Litman et al., 2025): VICReg + centering, EMA 0.996→0.9995
  - C-JEPA (Mo et al., NeurIPS 2024): VICReg integration prevents collapse
  - LeJEPA (Balestriero & LeCun, 2025): SIGReg replaces VICReg

Usage:
    cd /workspace/bdna-jepa
    python scripts/pretrain_ijepa.py \\
        --data data/processed/pretrain_2M.csv \\
        --tokenizer data/tokenizer/bpe_4096.json \\
        --epochs 30 --batch-size 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ═══════════════════════════════════════════════════════════════════════════
# BPE Tokenizer (from bdna-jepa v4)
# ═══════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """BPE tokenizer wrapping HuggingFace tokenizers library."""
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self._tokenizer.token_to_id("[PAD]") or 0
        self.mask_id = self._tokenizer.token_to_id("[MASK]") or 1
        self.unk_id = self._tokenizer.token_to_id("[UNK]") or 4

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, sequence: str) -> list[int]:
        return self._tokenizer.encode(sequence, add_special_tokens=False).ids

    def get_attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens != self.pad_id


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class GenomeFragmentDataset(Dataset):
    """CSV dataset with 'sequence' column, tokenized with BPE."""
    def __init__(self, csv_path: str, tokenizer: BPETokenizer, max_len: int = 512):
        df = pd.read_csv(csv_path)
        self.sequences = df["sequence"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.sequences[idx])
        ids = ids[:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# Encoder (scaled v3 SparseTransformerEncoder)
# ═══════════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """Transformer encoder returning pooled + per-token embeddings."""
    def __init__(self, vocab_size: int, embed_dim: int = 576, num_layers: int = 12,
                 num_heads: int = 9, ff_dim: int = 2304, max_seq_len: int = 512,
                 dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                              enable_nested_tensor=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None,
                return_token_embeddings: bool = False):
        B, L = tokens.shape
        if attention_mask is None:
            attention_mask = tokens != self.pad_token_id
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        x = self.dropout(x)
        key_padding_mask = ~attention_mask.bool()
        token_embs = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # Mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        info = {"attention_mask": attention_mask}
        if return_token_embeddings:
            info["token_embeddings"] = token_embs
        return pooled, info


# ═══════════════════════════════════════════════════════════════════════════
# I-JEPA Predictor (from v3, verified against I-JEPA paper)
# ═══════════════════════════════════════════════════════════════════════════

class _PredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class JEPAPredictor(nn.Module):
    """I-JEPA-style predictor: mask tokens + positional embed + self-attention.

    Takes context encoder token embeddings, replaces masked positions with
    learnable mask token (preserving positional information), runs self-attention
    so mask tokens attend to visible context, then projects back to encoder dim.

    This is THE key anti-collapse mechanism: the predictor must output DIFFERENT
    embeddings at different masked positions because of positional conditioning.
    """
    def __init__(self, embed_dim: int, predictor_dim: int = 384, depth: int = 4,
                 num_heads: int = 6, max_seq_len: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, predictor_dim), nn.LayerNorm(predictor_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            _PredictorBlock(predictor_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context_token_emb: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_token_emb: (B, L, D) from context encoder
            target_mask: (B, L) bool, True = masked/target position
        Returns:
            (B, L, D) predictions at ALL positions (loss selects masked ones)
        """
        B, L, _ = context_token_emb.shape
        x = self.input_proj(context_token_emb)  # (B, L, D_pred)
        # Replace masked positions with learnable mask token
        target_float = target_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = x * (1.0 - target_float) + self.mask_token.expand(B, L, -1) * target_float
        # Add positional embedding (critical for position-dependent predictions)
        x = x + self.pos_embed[:, :L, :]
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x))  # (B, L, D)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Block Masking (I-JEPA style, from v3)
# ═══════════════════════════════════════════════════════════════════════════

def multi_block_mask_1d(seq_len: int, mask_ratio: float, num_target_blocks: int,
                        min_block_len: int, eligible: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
    """I-JEPA multi-block masking for 1D sequences.

    Creates non-overlapping contiguous target spans. Context = everything outside targets.
    """
    B = eligible.shape[0]
    target_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    elig_lens = eligible.sum(dim=1)

    for b in range(B):
        L_elig = elig_lens[b].item()
        n_target = int(L_elig * mask_ratio)
        if L_elig < min_block_len * 2 or n_target < min_block_len:
            blen = max(1, min(min_block_len, int(L_elig) - 1))
            if blen > 0 and int(L_elig) > blen:
                start = random.randint(0, int(L_elig) - blen)
                target_mask[b, start:start + blen] = True
            continue

        per_block = max(min_block_len, n_target // num_target_blocks)
        remaining = n_target
        occupied = [False] * int(L_elig)

        for _ in range(num_target_blocks):
            if remaining < min_block_len:
                break
            blen = min(max(min_block_len, per_block), remaining, int(L_elig))
            placed = False
            for _attempt in range(50):
                max_start = int(L_elig) - blen
                if max_start < 0: break
                start = random.randint(0, max_start)
                if not any(occupied[start:start + blen]):
                    for i in range(start, start + blen):
                        occupied[i] = True
                    target_mask[b, start:start + blen] = True
                    remaining -= blen
                    placed = True
                    break

    target_mask &= eligible
    return target_mask


def curriculum_masking_params(epoch, total_epochs, mr_start=0.15, mr_end=0.50,
                              bl_start=3, bl_end=15):
    """Cosine curriculum: easy (low mask, small blocks) → hard."""
    progress = epoch / max(total_epochs - 1, 1)
    t = 0.5 * (1.0 - math.cos(math.pi * progress))
    mask_ratio = mr_start + t * (mr_end - mr_start)
    block_len = int(bl_start + t * (bl_end - bl_start))
    return mask_ratio, max(1, block_len)


# ═══════════════════════════════════════════════════════════════════════════
# VICReg (from C-JEPA — proven to prevent collapse in JEPA)
# ═══════════════════════════════════════════════════════════════════════════

def _variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(gamma - z.float().std(dim=0)).mean()

def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    N, D = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / max(N - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


# ═══════════════════════════════════════════════════════════════════════════
# Monitoring utilities
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rankme(embeddings: torch.Tensor) -> float:
    z = embeddings.float()
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + 1e-12)
    return torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()


# ═══════════════════════════════════════════════════════════════════════════
# B-JEPA Model (v3 architecture, v4 scale)
# ═══════════════════════════════════════════════════════════════════════════

class BJEPA(nn.Module):
    """B-JEPA: I-JEPA architecture for bacterial DNA sequences.

    Forward pass:
        1. Multi-block mask → target positions
        2. Context encoder: input with targets replaced by PAD → (B, L, D)
        3. Predictor: context embeddings + mask tokens at target positions → (B, L, D)
        4. Target encoder (EMA): full input → (B, L, D)
        5. Loss = MSE(pred[masked], target[masked]) + VICReg(context_pooled)
    """
    def __init__(self, encoder: TransformerEncoder, predictor_dim: int = 384,
                 predictor_depth: int = 4, predictor_heads: int = 6,
                 max_seq_len: int = 512, ema_start: float = 0.996, ema_end: float = 1.0):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        embed_dim = encoder.embed_dim
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, num_heads=predictor_heads,
            max_seq_len=max_seq_len,
        )
        self.ema_start = ema_start
        self.ema_end = ema_end
        self._ema_decay = ema_start

    def set_ema_decay(self, progress: float) -> float:
        t0, t1 = self.ema_start, self.ema_end
        self._ema_decay = t1 - (t1 - t0) * (1 + math.cos(math.pi * progress)) / 2
        return self._ema_decay

    @torch.no_grad()
    def update_ema(self):
        tau = self._ema_decay
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data.mul_(tau).add_(cp.data, alpha=1.0 - tau)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None,
                mask_ratio: float = 0.30, min_block_len: int = 5,
                num_target_blocks: int = 4, pad_token_id: int = 0):
        B, L = tokens.shape
        device = tokens.device

        if attention_mask is None:
            attention_mask = tokens != pad_token_id

        eligible = tokens != pad_token_id

        # I-JEPA multi-block masking
        target_mask = multi_block_mask_1d(
            L, mask_ratio, num_target_blocks, min_block_len, eligible, device,
        )

        # Context encoder: masked input
        masked_tokens = tokens.clone()
        masked_tokens[target_mask] = pad_token_id
        _, ctx_info = self.context_encoder(
            masked_tokens, attention_mask, return_token_embeddings=True,
        )
        ctx_emb = ctx_info["token_embeddings"]  # (B, L, D)

        # Predictor: context + mask tokens → predictions at all positions
        pred_all = self.predictor(ctx_emb, target_mask)  # (B, L, D)

        # Target encoder: full input (no grad)
        with torch.no_grad():
            _, tgt_info = self.target_encoder(
                tokens, attention_mask, return_token_embeddings=True,
            )
            tgt_emb = tgt_info["token_embeddings"]  # (B, L, D)

        # Extract predictions and targets at masked positions
        pred_masked = pred_all[target_mask]    # (N_masked, D)
        target_masked = tgt_emb[target_mask]   # (N_masked, D)

        # Context-pooled embedding (for VICReg regularization)
        visible = eligible & ~target_mask
        vis_float = visible.unsqueeze(-1).float()
        context_pooled = (ctx_emb * vis_float).sum(dim=1) / vis_float.sum(dim=1).clamp(min=1)

        info = {
            "target_mask": target_mask,
            "n_masked": target_mask.sum().item(),
            "context_pooled": context_pooled,
            "ema_decay": self._ema_decay,
        }
        return pred_masked, target_masked, info

    @torch.no_grad()
    def encode(self, tokens, attention_mask=None, use_target=True):
        enc = self.target_encoder if use_target else self.context_encoder
        pooled, _ = enc(tokens, attention_mask)
        return pooled


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def cosine_lr(optimizer, step, total, warmup, peak, floor=1e-6):
    if step < warmup:
        lr = peak * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        lr = floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def main():
    parser = argparse.ArgumentParser(description="B-JEPA v4.3 Pretraining")
    # Data
    parser.add_argument("--data", type=str, default="data/processed/pretrain_2M.csv")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/bpe_4096.json")
    parser.add_argument("--max-seq-len", type=int, default=512)
    # Encoder
    parser.add_argument("--embed-dim", type=int, default=576)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=9)
    parser.add_argument("--ff-dim", type=int, default=2304)
    # Predictor (I-JEPA proportions)
    parser.add_argument("--predictor-dim", type=int, default=384)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=6)
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    # EMA (I-JEPA defaults)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    # Masking (I-JEPA curriculum)
    parser.add_argument("--mask-ratio-start", type=float, default=0.15)
    parser.add_argument("--mask-ratio-end", type=float, default=0.50)
    parser.add_argument("--num-target-blocks", type=int, default=4)
    parser.add_argument("--min-block-start", type=int, default=3)
    parser.add_argument("--min-block-end", type=int, default=15)
    # Loss weights
    parser.add_argument("--vicreg-var-weight", type=float, default=25.0)
    parser.add_argument("--vicreg-cov-weight", type=float, default=1.0)
    # Logging
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/v4.3")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="bdna-jepa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tok = BPETokenizer(args.tokenizer)
    print(f"Tokenizer: BPE vocab={tok.vocab_size}")

    # Dataset
    dataset = GenomeFragmentDataset(args.data, tok, max_len=args.max_seq_len)
    print(f"Dataset: {len(dataset):,} sequences")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    steps_per_epoch = len(loader)

    # Model
    encoder = TransformerEncoder(
        vocab_size=tok.vocab_size, embed_dim=args.embed_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        ff_dim=args.ff_dim, max_seq_len=args.max_seq_len,
        pad_token_id=tok.pad_id,
    )
    model = BJEPA(
        encoder, predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        max_seq_len=args.max_seq_len, ema_start=args.ema_start, ema_end=args.ema_end,
    ).to(device)

    n_enc = sum(p.numel() for p in model.context_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"Encoder: {n_enc/1e6:.1f}M params ({args.num_layers}L x {args.embed_dim}D)")
    print(f"Predictor: {n_pred/1e6:.1f}M params ({args.predictor_depth}L x {args.predictor_dim}D)")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.context_encoder.parameters(), "lr": args.lr},
        {"params": model.predictor.parameters(), "lr": args.lr},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # W&B
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"bjepa-v4.3-{args.num_layers}L{args.embed_dim}D-ijepa",
            config=vars(args),
        )

    print(f"\n{'='*70}")
    print(f"  B-JEPA v4.3 — I-JEPA Architecture for Bacterial DNA")
    print(f"  {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    print(f"  Masking: {args.mask_ratio_start:.0%}→{args.mask_ratio_end:.0%} (4 blocks)")
    print(f"  EMA: {args.ema_start}→{args.ema_end}")
    print(f"  VICReg: var={args.vicreg_var_weight}, cov={args.vicreg_cov_weight}")
    print(f"{'='*70}\n")

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()

        # Curriculum masking
        progress = epoch / max(args.epochs - 1, 1)
        ema_decay = model.set_ema_decay(progress)
        mask_ratio, min_block_len = curriculum_masking_params(
            epoch, args.epochs,
            args.mask_ratio_start, args.mask_ratio_end,
            args.min_block_start, args.min_block_end,
        )

        epoch_losses = []
        epoch_pred_mse = []
        epoch_vicreg_var = []
        epoch_cos_sim = []

        optimizer.zero_grad(set_to_none=True)

        for step, tokens in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                                           leave=True, ncols=120), start=1):
            global_step += 1
            lr = cosine_lr(optimizer, global_step, total_steps, warmup_steps, args.lr, args.min_lr)

            tokens = tokens.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Forward: per-token prediction at masked positions
                pred, target, info = model(
                    tokens, mask_ratio=mask_ratio, min_block_len=min_block_len,
                    num_target_blocks=args.num_target_blocks, pad_token_id=tok.pad_id,
                )

                # Loss 1: Per-token prediction MSE (I-JEPA primary loss)
                pred_loss = F.mse_loss(pred, target)

                # Loss 2: VICReg on context-pooled embeddings (C-JEPA anti-collapse)
                ctx_pooled = info["context_pooled"]
                var_loss = _variance_loss(ctx_pooled)
                cov_loss = _covariance_loss(ctx_pooled)

                total_loss = pred_loss + args.vicreg_var_weight * var_loss + args.vicreg_cov_weight * cov_loss

                scaled = total_loss / args.grad_accum

            scaler.scale(scaled).backward()

            if step % args.grad_accum == 0 or step == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # EMA update
            model.update_ema()

            # Metrics
            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred, target, dim=-1).mean().item()

            epoch_losses.append(total_loss.item())
            epoch_pred_mse.append(pred_loss.item())
            epoch_vicreg_var.append(var_loss.item())
            epoch_cos_sim.append(cos_sim)

            # Step logging
            if global_step % args.log_every == 0:
                msg = (f"Step {global_step} | epoch {epoch} | loss={total_loss.item():.4f} | "
                       f"pred={pred_loss.item():.4f} | cos={cos_sim:.3f} | "
                       f"var={var_loss.item():.4f} | mr={mask_ratio:.2f} | lr={lr:.2e}")
                print(f"  {msg}")

                if use_wandb:
                    wandb.log({
                        "step/loss": total_loss.item(),
                        "step/pred_mse": pred_loss.item(),
                        "step/cos_sim": cos_sim,
                        "step/vicreg_var": var_loss.item(),
                        "step/vicreg_cov": cov_loss.item(),
                        "step/lr": lr,
                        "step/ema_decay": ema_decay,
                        "step/mask_ratio": mask_ratio,
                        "step/n_masked": info["n_masked"],
                    }, step=global_step)

        # Epoch summary
        ep_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses)
        avg_pred = np.mean(epoch_pred_mse)
        avg_cos = np.mean(epoch_cos_sim)
        avg_var = np.mean(epoch_vicreg_var)

        print(f"\nEpoch {epoch+1}/{args.epochs} complete | avg_loss={avg_loss:.4f} | "
              f"pred={avg_pred:.4f} | cos={avg_cos:.3f} | var={avg_var:.4f} | "
              f"time={ep_time:.1f}s")

        # Eval
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_embs = []
                for i, batch in enumerate(loader):
                    if i >= 50: break
                    batch = batch.to(device)
                    emb = model.encode(batch)
                    eval_embs.append(emb.cpu())
                eval_embs = torch.cat(eval_embs, dim=0)
                rankme = compute_rankme(eval_embs)
                std = eval_embs.std().item()

            print(f"  Eval | RankMe={rankme:.1f} | std={std:.4f}")

            if use_wandb:
                wandb.log({
                    "eval/rankme": rankme,
                    "eval/std": std,
                    "epoch/avg_loss": avg_loss,
                    "epoch/avg_pred_mse": avg_pred,
                    "epoch/avg_cos_sim": avg_cos,
                }, step=global_step)

            # RED FLAG checks
            if rankme < 50:
                print(f"  ⚠ WARNING: RankMe={rankme:.1f} < 50 — possible collapse!")
            if std < 0.1:
                print(f"  ⚠ WARNING: std={std:.4f} < 0.1 — severe collapse!")
            if std > 5.0:
                print(f"  ⚠ WARNING: std={std:.4f} > 5.0 — norm explosion!")

        # Save
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch{epoch+1:04d}.pt")
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": model.context_encoder.state_dict(),
                "target_encoder_state_dict": model.target_encoder.state_dict(),
                "predictor_state_dict": model.predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "avg_loss": avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    torch.save({
        "epoch": args.epochs,
        "encoder_state_dict": model.context_encoder.state_dict(),
        "target_encoder_state_dict": model.target_encoder.state_dict(),
        "predictor_state_dict": model.predictor.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"\nTraining complete. Final model: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
