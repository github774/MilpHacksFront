#!/usr/bin/env python3
"""Shared data utilities for the persona reaction/emotion models."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

EMB_DIM = 384
REACTION_KEYS = ["neutral", "like", "dislike", "like_share", "dislike_share"]
EMOTION_KEYS = ["empathy", "relation", "inspiration", "curiosity", "joy"]


class TranscriptDataset(Dataset):
    """One example = a transcript (sequence of segment embeddings) + its labels."""

    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seg = np.array(
                    [s["embedding"] for s in rec["segments"]], dtype=np.float32
                )
                reactions = np.array(
                    [rec["reactions"][k] for k in REACTION_KEYS], dtype=np.float32
                )
                emotions = np.array(
                    [rec["emotions"][k] for k in EMOTION_KEYS], dtype=np.float32
                )
                self.records.append((seg, reactions, emotions))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_personas(csv_path: str) -> np.ndarray:
    vectors = []
    with open(csv_path, encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            _, emb = line.split(",", 1)
            emb = emb.strip().strip('"')
            vectors.append(json.loads(emb))
    return np.asarray(vectors, dtype=np.float32)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
