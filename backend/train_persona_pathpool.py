#!/usr/bin/env python3
"""Persona-conditioned path-pool reaction/emotion model.

Why persona-conditioned TARGETS exist
-------------------------------------
The raw dataset labels (reactions / emotions) are aggregate-audience: they do
not depend on who is watching, so a persona input would simply be ignored by the
optimizer. Transcript segments and the Nemotron persona vectors live in the SAME
bge-small embedding space, so cosine(persona, transcript) is a meaningful
semantic affinity. We use a standardized affinity s to tilt each transcript's
base labels:
  - reactions: shift probability toward like / like_share and away from
    dislike / dislike_share by alpha * s (in logit space).
  - emotions: nudge empathy / relation / joy by beta * s.
Because s is standardized to ~zero mean per population, averaging the conditioned
labels over personas reproduces the original aggregate labels - we are only
adding the persona-dependent variation that was missing.

Architecture (persona-conditioned path-pool)
--------------------------------------------
  token x (384), persona p (384)
  1. per-channel path nonlinearity on each token  -> z          (384 unique phi's)
  2. persona-conditioned attention pool: weight head sees
     [z, z * p] per token -> scalar -> softmax over tokens -> pooled u
  3. persona interaction at the head: head input = [u, u * p]
  4. reaction head (softmax) + emotion head (sigmoid)
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
    load_personas,
    soft_cross_entropy,
)
from paths import PERSONA_EMBEDDINGS_CSV, PERSONA_WEIGHTS, TRAINING_EMBEDDED_JSONL

# REACTION_KEYS = [neutral, like, dislike, like_share, dislike_share]
# EMOTION_KEYS  = [empathy, relation, inspiration, curiosity, joy]
LIKE_IDX = [1, 3]      # like, like_share
DISLIKE_IDX = [2, 4]   # dislike, dislike_share
EMO_POS_IDX = [0, 1, 4]  # empathy, relation, joy


class PerChannelPath(nn.Module):
    """fullpath transform with 7 learned scalars per channel (unique per row)."""

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
        u = torch.clamp(self.b * x, -1.0 + self.eps, 1.0 - self.eps)
        arccos_u = torch.atan2(torch.sqrt(1.0 - u * u), u)
        return (
            arccos_u
            + self.c * torch.sin(self.d * x) ** 3
            + self.f * x
            + self.g * (x - self.h) ** 2
            + self.m
        )


class PersonaPathPool(nn.Module):
    def __init__(self, dim: int = EMB_DIM, dropout: float = 0.3):
        super().__init__()
        self.path = PerChannelPath(dim)
        # persona-conditioned pooling: sees [z, z*p] -> scalar weight per token
        self.weight_head = nn.Linear(2 * dim, 1)
        # separate norms: the path-space content half (values ~pi/2 from arccos)
        # would otherwise dominate a shared LayerNorm and crush the much smaller
        # raw persona-interaction half.
        self.norm_z = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        # heads see [pooled, pooled * persona]
        self.reaction_head = nn.Linear(2 * dim, len(REACTION_KEYS))
        self.emotion_head = nn.Linear(2 * dim, len(EMOTION_KEYS))

    def forward(self, seg, persona, mask):
        # seg: (B, N, dim), persona: (B, dim), mask: (B, N)
        z = self.path(seg)                           # path-transformed tokens
        p = persona.unsqueeze(1)                      # (B, 1, dim)
        # persona-conditioned pooling. The persona interaction is computed in the
        # RAW embedding space (seg * p), where the affinity / cosine actually
        # lives; z (path space) carries transcript content.
        w_in = torch.cat([z, seg * p], dim=-1)        # (B, N, 2*dim)
        logits = self.weight_head(w_in).squeeze(-1)   # (B, N)
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)           # pooling weights
        a = attn.unsqueeze(-1)
        pooled_z = (a * z).sum(dim=1)                  # content (path space)
        pooled_raw = (a * seg).sum(dim=1)              # content (raw space)
        # raw interaction with persona -> linear head can sum it to recover affinity
        inter = pooled_raw * persona
        feat = torch.cat([self.norm_z(pooled_z), self.norm_i(inter)], dim=-1)
        feat = self.drop(feat)
        return self.reaction_head(feat), self.emotion_head(feat), attn


# --------------------------------------------------------------------------- #
# Persona-conditioned target generation
# --------------------------------------------------------------------------- #
def transcript_mean_embeddings(ds: TranscriptDataset) -> np.ndarray:
    embs = []
    for seg, _, _ in ds.records:
        v = seg.mean(axis=0)
        v = v / max(np.linalg.norm(v), 1e-9)
        embs.append(v)
    return np.asarray(embs, dtype=np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


def condition_labels(base_react, base_emo, s, alpha, beta):
    """Tilt base labels by standardized affinity s. Vectorized over the batch.
    base_react: (B,5), base_emo: (B,5), s: (B,)."""
    logits = np.log(base_react + 1e-6)
    shift = (alpha * s)[:, None]
    logits[:, LIKE_IDX] += shift
    logits[:, DISLIKE_IDX] -= shift
    react = softmax_np(logits)
    emo = base_emo.copy()
    emo[:, EMO_POS_IDX] = np.clip(emo[:, EMO_POS_IDX] + beta * s[:, None], 0.0, 1.0)
    return react.astype(np.float32), emo.astype(np.float32)


def make_collate(personas, tmean, aff_mean, aff_std, rng, alpha, beta,
                 assigned_pidx=None, deterministic=False):
    n_personas = personas.shape[0]

    def collate(items):
        # items: list of (record_index, (seg, base_react, base_emo))
        idxs = [it[0] for it in items]
        recs = [it[1] for it in items]
        B = len(recs)
        max_n = max(seg.shape[0] for seg, _, _ in recs)
        seg_t = torch.zeros(B, max_n, EMB_DIM)
        mask = torch.zeros(B, max_n, dtype=torch.bool)
        base_r = np.zeros((B, len(REACTION_KEYS)), dtype=np.float32)
        base_e = np.zeros((B, len(EMOTION_KEYS)), dtype=np.float32)
        for i, (seg, r, e) in enumerate(recs):
            n = seg.shape[0]
            seg_t[i, :n] = torch.from_numpy(seg)
            mask[i, :n] = True
            base_r[i] = r
            base_e[i] = e

        if deterministic:
            # fixed persona per transcript -> fixed conditioned labels (for a
            # clean, repeatable validation / ablation comparison).
            pidx = assigned_pidx[np.asarray(idxs)]
        else:
            pidx = rng.integers(0, n_personas, size=B)  # training augmentation
        persona = personas[pidx]                       # (B, dim)
        aff = (tmean[idxs] * persona).sum(axis=1)       # cosine (both unit-norm)
        s = (aff - aff_mean) / aff_std                  # standardized affinity
        react, emo = condition_labels(base_r, base_e, s, alpha, beta)

        return (
            seg_t,
            torch.from_numpy(persona),
            mask,
            torch.from_numpy(react),
            torch.from_numpy(emo),
            torch.from_numpy(s.astype(np.float32)),
        )

    return collate


class IndexedSubset(torch.utils.data.Dataset):
    """Yields (global_record_index, record) so collate can look up affinity."""

    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gi = self.indices[i]
        return gi, self.ds.records[gi]


def evaluate(model, loader, device, persona_override=None):
    model.eval()
    tot_r = tot_e = n = 0.0
    with torch.no_grad():
        for seg, persona, mask, react, emo, _ in loader:
            if persona_override == "shuffle":
                g = torch.Generator().manual_seed(1234)
                persona = persona[torch.randperm(persona.size(0), generator=g)]
            elif persona_override == "zero":
                persona = torch.zeros_like(persona)
            seg, persona, mask = seg.to(device), persona.to(device), mask.to(device)
            react, emo = react.to(device), emo.to(device)
            r_logits, e_logits, _ = model(seg, persona, mask)
            tot_r += soft_cross_entropy(r_logits, react).item() * seg.size(0)
            tot_e += F.binary_cross_entropy_with_logits(e_logits, emo).item() * seg.size(0)
            n += seg.size(0)
    return tot_r / n, tot_e / n


def count_parameters(model):
    out = {
        "per_channel_path": sum(p.numel() for p in model.path.parameters()),
        "pool_weight_head": sum(p.numel() for p in model.weight_head.parameters()),
        "norm": sum(p.numel() for p in model.norm_z.parameters())
        + sum(p.numel() for p in model.norm_i.parameters()),
        "reaction_head": sum(p.numel() for p in model.reaction_head.parameters()),
        "emotion_head": sum(p.numel() for p in model.emotion_head.parameters()),
    }
    out["TOTAL"] = sum(p.numel() for p in model.parameters())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(TRAINING_EMBEDDED_JSONL))
    ap.add_argument("--personas", default=str(PERSONA_EMBEDDINGS_CSV))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="reaction tilt strength per std of affinity")
    ap.add_argument("--beta", type=float, default=0.08,
                    help="emotion tilt strength per std of affinity")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--review-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PersonaPathPool(dropout=args.dropout).to(device)
    print(f"device: {device}")
    counts = count_parameters(model)
    print("=" * 60)
    print("PERSONA PATH-POOL")
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

    print("Loading personas + transcripts...")
    personas = load_personas(args.personas)
    personas = personas / np.maximum(
        np.linalg.norm(personas, axis=1, keepdims=True), 1e-9
    )
    ds = TranscriptDataset(args.data)
    tmean = transcript_mean_embeddings(ds)  # (n_records, dim), unit-norm

    # affinity stats over a random sample of (transcript, persona) pairs
    sp = rng.integers(0, personas.shape[0], size=4000)
    st = rng.integers(0, tmean.shape[0], size=4000)
    sample_aff = (tmean[st] * personas[sp]).sum(axis=1)
    aff_mean, aff_std = float(sample_aff.mean()), float(sample_aff.std() + 1e-9)
    print(f"affinity mean={aff_mean:.4f}  std={aff_std:.4f}")

    n = len(ds)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = IndexedSubset(ds, train_idx)
    val_ds = IndexedSubset(ds, val_idx)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    # fixed persona assignment per record for deterministic validation
    assigned_pidx = np.random.default_rng(args.seed + 1).integers(
        0, personas.shape[0], size=n
    )

    train_collate = make_collate(personas, tmean, aff_mean, aff_std, rng,
                                 args.alpha, args.beta)
    val_collate = make_collate(personas, tmean, aff_mean, aff_std, rng,
                               args.alpha, args.beta,
                               assigned_pidx=assigned_pidx, deterministic=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=val_collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for seg, persona, mask, react, emo, _ in train_loader:
            seg, persona, mask = seg.to(device), persona.to(device), mask.to(device)
            react, emo = react.to(device), emo.to(device)
            r_logits, e_logits, _ = model(seg, persona, mask)
            loss = (soft_cross_entropy(r_logits, react)
                    + F.binary_cross_entropy_with_logits(e_logits, emo))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vr, ve = evaluate(model, val_loader, device)
        val = vr + ve
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val_react_CE {vr:.4f}  val_emo_BCE {ve:.4f}"
                  f"   (best {best:.4f}, bad {bad})")
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("-" * 60)
    print("ABLATION: does the model actually USE the persona?")
    vr, ve = evaluate(model, val_loader, device)
    print(f"  correct persona : react_CE {vr:.4f}  emo_BCE {ve:.4f}")
    sr, se = evaluate(model, val_loader, device, persona_override="shuffle")
    print(f"  shuffled persona: react_CE {sr:.4f}  emo_BCE {se:.4f}  "
          f"(should be worse)")
    zr, ze = evaluate(model, val_loader, device, persona_override="zero")
    print(f"  zero persona    : react_CE {zr:.4f}  emo_BCE {ze:.4f}")

    torch.save(model.state_dict(), PERSONA_WEIGHTS)
    print(f"Saved {PERSONA_WEIGHTS}")


if __name__ == "__main__":
    main()
