#!/usr/bin/env python3
"""Path-pool reaction/emotion classifier.

Architecture (per the design):

  Input: a transcript = sequence of N segment embeddings, each 384-d.

  1. Per-channel path nonlinearity on every token embedding. There are 384
     independent path transforms (one per embedding dimension / "row"), each
     with its own 7 learned scalars (b, c, d, f, g, h, m):
         u_i   = clip(b_i * x_i, -1+eps, 1-eps)
         phi_i = arccos(u_i) + c_i*sin(d_i*x_i)^3 + f_i*x_i + g_i*(x_i-h_i)^2 + m_i
     arccos(u) is evaluated as atan2(sqrt(1-u^2), u) on the clipped domain.

  2. Attention pooling (NOT mean pooling): a small head maps each token's
     transformed embedding to a scalar; softmax over the N tokens gives pooling
     weights; the pooled vector is their weighted sum.

  3. Two heads on the pooled vector:
       reaction head: 384 -> 5  softmax  (neutral, like, dislike, like_share, dislike_share)
       emotion  head: 384 -> 5  sigmoid  (empathy, relation, inspiration, curiosity, joy)

Small dataset (~900 train) so the model is deliberately tiny and regularized
(dropout + weight decay + early stopping).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    EMB_DIM,
    EMOTION_KEYS,
    REACTION_KEYS,
    TranscriptDataset,
    soft_cross_entropy,
)
from paths import PATHPOOL_WEIGHTS, TRAINING_EMBEDDED_JSONL


class PerChannelPath(nn.Module):
    """fullpath transform with 7 learned scalars PER channel (unique per row)."""

    def __init__(self, dim: int = EMB_DIM, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.b = nn.Parameter(torch.ones(dim))
        self.c = nn.Parameter(torch.zeros(dim))
        self.d = nn.Parameter(torch.ones(dim))
        self.f = nn.Parameter(torch.ones(dim))
        self.g = nn.Parameter(torch.zeros(dim))
        self.h = nn.Parameter(torch.zeros(dim))
        self.m = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim); each scalar broadcasts over the leading dims.
        u = torch.clamp(self.b * x, -1.0 + self.eps, 1.0 - self.eps)
        arccos_u = torch.atan2(torch.sqrt(1.0 - u * u), u)
        return (
            arccos_u
            + self.c * torch.sin(self.d * x) ** 3
            + self.f * x
            + self.g * (x - self.h) ** 2
            + self.m
        )


class PathPoolClassifier(nn.Module):
    def __init__(self, dim: int = EMB_DIM, dropout: float = 0.3):
        super().__init__()
        self.path = PerChannelPath(dim)
        self.weight_head = nn.Linear(dim, 1)  # small head: token -> scalar
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.reaction_head = nn.Linear(dim, len(REACTION_KEYS))
        self.emotion_head = nn.Linear(dim, len(EMOTION_KEYS))

    def forward(self, seg, mask):
        # seg: (B, N, dim), mask: (B, N) True = real token
        z = self.path(seg)  # per-channel path nonlinearity
        logits = self.weight_head(z).squeeze(-1)  # (B, N)
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)  # pooling weights over tokens
        pooled = (attn.unsqueeze(-1) * z).sum(dim=1)  # (B, dim)
        pooled = self.drop(self.norm(pooled))
        return self.reaction_head(pooled), self.emotion_head(pooled), attn


def collate(batch):
    max_n = max(seg.shape[0] for seg, _, _ in batch)
    B = len(batch)
    seg_t = torch.zeros(B, max_n, EMB_DIM)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    react_t = torch.zeros(B, len(REACTION_KEYS))
    emo_t = torch.zeros(B, len(EMOTION_KEYS))
    for i, (seg, reactions, emotions) in enumerate(batch):
        n = seg.shape[0]
        seg_t[i, :n] = torch.from_numpy(seg)
        mask[i, :n] = True
        react_t[i] = torch.from_numpy(reactions)
        emo_t[i] = torch.from_numpy(emotions)
    return seg_t, mask, react_t, emo_t


def evaluate(model, loader, device):
    model.eval()
    tot_r = tot_e = n = 0.0
    with torch.no_grad():
        for seg, mask, react, emo in loader:
            seg, mask, react, emo = (
                seg.to(device), mask.to(device), react.to(device), emo.to(device)
            )
            r_logits, e_logits, _ = model(seg, mask)
            tot_r += soft_cross_entropy(r_logits, react).item() * seg.size(0)
            tot_e += F.binary_cross_entropy_with_logits(e_logits, emo).item() * seg.size(0)
            n += seg.size(0)
    return tot_r / n, tot_e / n


def count_parameters(model):
    out = {
        "per_channel_path": sum(p.numel() for p in model.path.parameters()),
        "pool_weight_head": sum(p.numel() for p in model.weight_head.parameters()),
        "norm": sum(p.numel() for p in model.norm.parameters()),
        "reaction_head": sum(p.numel() for p in model.reaction_head.parameters()),
        "emotion_head": sum(p.numel() for p in model.emotion_head.parameters()),
    }
    out["TOTAL"] = sum(p.numel() for p in model.parameters())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(TRAINING_EMBEDDED_JSONL))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--review-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PathPoolClassifier(dropout=args.dropout).to(device)
    print(f"device: {device}")

    counts = count_parameters(model)
    print("=" * 60)
    print("PATH-POOL CLASSIFIER")
    print("=" * 60)
    print(model)
    print("-" * 60)
    for k, v in counts.items():
        if k == "TOTAL":
            print("-" * 60)
        print(f"  {k:20s} {v:>10,d}")
    print("=" * 60)
    if args.review_only:
        return

    ds = TranscriptDataset(args.data)
    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    print(f"train={n_train}  val={n_val}")
    print("prior baselines:  react_CE=1.3657  emo_BCE=0.6817")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for seg, mask, react, emo in train_loader:
            seg, mask, react, emo = (
                seg.to(device), mask.to(device), react.to(device), emo.to(device)
            )
            r_logits, e_logits, _ = model(seg, mask)
            loss = (soft_cross_entropy(r_logits, react)
                    + F.binary_cross_entropy_with_logits(e_logits, emo))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vr, ve = evaluate(model, val_loader, device)
        val = vr + ve
        if val < best - 1e-4:
            best, best_state, bad = val, {k: v.detach().cpu().clone()
                                          for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val_react_CE {vr:.4f}  val_emo_BCE {ve:.4f}"
                  f"   (best sum {best:.4f}, bad {bad})")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    vr, ve = evaluate(model, val_loader, device)
    print(f"BEST  val_react_CE {vr:.4f}  val_emo_BCE {ve:.4f}")
    torch.save(model.state_dict(), PATHPOOL_WEIGHTS)
    print(f"Saved {PATHPOOL_WEIGHTS}")


if __name__ == "__main__":
    main()
