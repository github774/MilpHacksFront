#!/usr/bin/env python3
"""Inference pipeline: transcript + persona text -> embeddings -> predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from common import EMOTION_KEYS, EMB_DIM, REACTION_KEYS
from paths import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_SHARPNESS,
    PERSONA_CSV,
    PERSONA_EMBEDDINGS_CSV,
    PERSONA_WEIGHTS,
    TRAINING_JSONL,
)
from train_persona_pathpool import PersonaPathPool

DEFAULT_WEIGHTS = PERSONA_WEIGHTS
DEFAULT_PERSONAS_CSV = PERSONA_EMBEDDINGS_CSV
# Population stats from train_persona_pathpool affinity sampling (seed=7).
AFFINITY_MEAN = 0.5511
AFFINITY_STD = 0.0389
NEUTRAL_IDX = REACTION_KEYS.index("neutral")
LIKE_IDX = [REACTION_KEYS.index("like"), REACTION_KEYS.index("like_share")]
DISLIKE_IDX = [REACTION_KEYS.index("dislike"), REACTION_KEYS.index("dislike_share")]
EMO_POS_IDX = [
    EMOTION_KEYS.index("empathy"),
    EMOTION_KEYS.index("relation"),
    EMOTION_KEYS.index("joy"),
]

_mean_persona_cache: np.ndarray | None = None


def get_mean_persona(
    personas_csv: str | Path = DEFAULT_PERSONAS_CSV,
) -> np.ndarray:
    """Average Nemotron persona embedding (unit-norm), cached."""
    global _mean_persona_cache
    if _mean_persona_cache is not None:
        return _mean_persona_cache
    from common import load_personas

    personas = load_personas(str(personas_csv))
    personas = personas / np.maximum(
        np.linalg.norm(personas, axis=1, keepdims=True), 1e-9
    )
    mean = personas.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-9)
    _mean_persona_cache = mean.astype(np.float32)
    return _mean_persona_cache


def embed_texts(
    texts: Sequence[str],
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed one or more text strings with bge-small (L2-normalized)."""
    if not texts:
        raise ValueError("texts must be non-empty")
    if model is None:
        model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_transcript_segments(
    segment_texts: Sequence[str],
    model: SentenceTransformer | None = None,
    **kwargs,
) -> np.ndarray:
    """Return (N, 384) segment embeddings for a transcript."""
    cleaned = [str(t).strip() for t in segment_texts]
    if not cleaned or any(not t for t in cleaned):
        raise ValueError("every segment must have non-empty text")
    return embed_texts(cleaned, model=model, **kwargs)


def embed_persona_text(
    persona_text: str,
    model: SentenceTransformer | None = None,
    **kwargs,
) -> np.ndarray:
    """Return a single (384,) persona embedding."""
    text = persona_text.strip()
    if not text:
        raise ValueError("persona_text must be non-empty")
    return embed_texts([text], model=model, **kwargs)[0]


def load_pathpool(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    device: str | torch.device | None = None,
    dropout: float = 0.0,
) -> tuple[PersonaPathPool, torch.device]:
    """Load PersonaPathPool weights for inference (dropout disabled)."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights not found: {path}")

    model = PersonaPathPool(dropout=dropout)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device


def _standardized_affinity(
    segment_embeddings: np.ndarray, persona_embedding: np.ndarray
) -> tuple[float, float]:
    mean = segment_embeddings.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-9)
    persona = persona_embedding / max(float(np.linalg.norm(persona_embedding)), 1e-9)
    affinity = float(np.dot(mean, persona))
    s = (affinity - AFFINITY_MEAN) / AFFINITY_STD
    return affinity, s


def sharpen_logits(
    persona_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    sharpness: float,
    *,
    affinity_s: float | None = None,
    mismatch_neutral_gain: float = 1.2,
    mismatch_emotion_gain: float = 0.8,
    is_reaction: bool = True,
) -> torch.Tensor:
    """Amplify persona-specific logit delta vs an average-persona baseline."""
    if sharpness == 1.0 and (affinity_s is None or affinity_s >= 0):
        return persona_logits
    out = baseline_logits + sharpness * (persona_logits - baseline_logits)
    if affinity_s is not None and affinity_s < 0:
        gap = -affinity_s
        out = out.clone()
        if is_reaction:
            out[..., NEUTRAL_IDX] = out[..., NEUTRAL_IDX] + mismatch_neutral_gain * gap
            for idx in LIKE_IDX + DISLIKE_IDX:
                out[..., idx] = out[..., idx] - mismatch_neutral_gain * 0.45 * gap
        else:
            for idx in EMO_POS_IDX:
                out[..., idx] = out[..., idx] - mismatch_emotion_gain * gap
    return out


def predict(
    segment_embeddings: np.ndarray,
    persona_embedding: np.ndarray,
    model: PersonaPathPool,
    device: torch.device | None = None,
    *,
    sharpness: float = 1.0,
    mismatch_neutral_gain: float = 1.2,
    mismatch_emotion_gain: float = 0.8,
    mean_persona: np.ndarray | None = None,
) -> dict:
    """Run persona_pathpool on one transcript + persona.

    Returns reaction probabilities (softmax), emotion probabilities (sigmoid),
    and raw attention weights over segments.

    When sharpness > 1, logits are amplified relative to a zero-persona baseline
    so persona effects show up more clearly in printed results.
    """
    seg = np.asarray(segment_embeddings, dtype=np.float32)
    persona = np.asarray(persona_embedding, dtype=np.float32)
    if seg.ndim != 2 or seg.shape[1] != EMB_DIM:
        raise ValueError(f"segment_embeddings must be (N, {EMB_DIM})")
    if persona.shape != (EMB_DIM,):
        raise ValueError(f"persona_embedding must be ({EMB_DIM},)")

    if device is None:
        device = next(model.parameters()).device

    seg_t = torch.from_numpy(seg).unsqueeze(0).to(device)
    persona_t = torch.from_numpy(persona).unsqueeze(0).to(device)
    mask = torch.ones(1, seg.shape[0], dtype=torch.bool, device=device)
    affinity, affinity_s = _standardized_affinity(seg, persona)

    if mean_persona is None:
        mean_persona = get_mean_persona()
    mean_t = torch.from_numpy(mean_persona).unsqueeze(0).to(device)

    with torch.no_grad():
        r_logits, e_logits, attn = model(seg_t, persona_t, mask)
        if sharpness != 1.0:
            r_base, e_base, _ = model(seg_t, mean_t, mask)
            r_logits = sharpen_logits(
                r_logits,
                r_base,
                sharpness,
                affinity_s=affinity_s,
                mismatch_neutral_gain=mismatch_neutral_gain,
                is_reaction=True,
            )
            e_logits = sharpen_logits(
                e_logits,
                e_base,
                sharpness,
                affinity_s=affinity_s,
                mismatch_emotion_gain=mismatch_emotion_gain,
                is_reaction=False,
            )
        reactions = F.softmax(r_logits, dim=-1)[0].cpu().numpy()
        emotions = torch.sigmoid(e_logits)[0].cpu().numpy()
        weights = attn[0].cpu().numpy()

    return {
        "reactions": {k: float(reactions[i]) for i, k in enumerate(REACTION_KEYS)},
        "emotions": {k: float(emotions[i]) for i, k in enumerate(EMOTION_KEYS)},
        "segment_weights": weights.tolist(),
        "affinity": affinity,
        "affinity_s": affinity_s,
        "sharpened": sharpness != 1.0,
    }


def predict_batch(
    segment_embeddings: np.ndarray,
    persona_embeddings: np.ndarray,
    model: PersonaPathPool,
    device: torch.device | None = None,
    *,
    sharpness: float = 1.0,
    mismatch_neutral_gain: float = 1.2,
    mismatch_emotion_gain: float = 0.8,
    mean_persona: np.ndarray | None = None,
) -> list[dict]:
    """Run persona_pathpool on one transcript against many personas (batched)."""
    seg = np.asarray(segment_embeddings, dtype=np.float32)
    personas = np.asarray(persona_embeddings, dtype=np.float32)
    if seg.ndim != 2 or seg.shape[1] != EMB_DIM:
        raise ValueError(f"segment_embeddings must be (N, {EMB_DIM})")
    if personas.ndim != 2 or personas.shape[1] != EMB_DIM:
        raise ValueError(f"persona_embeddings must be (B, {EMB_DIM})")

    if device is None:
        device = next(model.parameters()).device

    b = personas.shape[0]
    n = seg.shape[0]
    seg_t = (
        torch.from_numpy(seg)
        .unsqueeze(0)
        .expand(b, n, EMB_DIM)
        .contiguous()
        .to(device)
    )
    persona_t = torch.from_numpy(personas).to(device)
    mask = torch.ones(b, n, dtype=torch.bool, device=device)

    tmean = seg.mean(axis=0)
    tmean = tmean / max(float(np.linalg.norm(tmean)), 1e-9)
    affinities = personas @ tmean
    affinity_s_np = (affinities - AFFINITY_MEAN) / AFFINITY_STD
    affinity_s = torch.from_numpy(affinity_s_np.astype(np.float32)).to(device)

    if mean_persona is None:
        mean_persona = get_mean_persona()
    mean_t = torch.from_numpy(mean_persona).unsqueeze(0).expand(b, -1).to(device)

    with torch.no_grad():
        r_logits, e_logits, attn = model(seg_t, persona_t, mask)
        if sharpness != 1.0:
            r_base, e_base, _ = model(seg_t, mean_t, mask)
            r_logits = r_base + sharpness * (r_logits - r_base)
            e_logits = e_base + sharpness * (e_logits - e_base)
            neg = affinity_s < 0
            if bool(neg.any()):
                gap = torch.where(neg, -affinity_s, torch.zeros_like(affinity_s))
                r_logits = r_logits.clone()
                e_logits = e_logits.clone()
                r_logits[neg, NEUTRAL_IDX] += mismatch_neutral_gain * gap[neg]
                for idx in LIKE_IDX + DISLIKE_IDX:
                    r_logits[neg, idx] -= mismatch_neutral_gain * 0.45 * gap[neg]
                for idx in EMO_POS_IDX:
                    e_logits[neg, idx] -= mismatch_emotion_gain * gap[neg]
        reactions = F.softmax(r_logits, dim=-1).cpu().numpy()
        emotions = torch.sigmoid(e_logits).cpu().numpy()
        weights = attn.cpu().numpy()

    out = []
    for i in range(b):
        out.append(
            {
                "reactions": {
                    k: float(reactions[i, j]) for j, k in enumerate(REACTION_KEYS)
                },
                "emotions": {
                    k: float(emotions[i, j]) for j, k in enumerate(EMOTION_KEYS)
                },
                "segment_weights": weights[i].tolist(),
                "affinity": float(affinities[i]),
                "affinity_s": float(affinity_s_np[i]),
                "sharpened": sharpness != 1.0,
            }
        )
    return out


def run_pipeline(
    segment_texts: Sequence[str],
    persona_text: str,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    embed_model: SentenceTransformer | None = None,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    device: str | torch.device | None = None,
    *,
    sharpness: float = DEFAULT_SHARPNESS,
    mismatch_neutral_gain: float = 1.2,
    mismatch_emotion_gain: float = 0.8,
) -> dict:
    """Full path: raw text -> embeddings -> persona_pathpool prediction."""
    model_st = embed_model
    if model_st is None:
        model_st = SentenceTransformer(embed_model_name, device=device)

    seg_emb = embed_transcript_segments(segment_texts, model=model_st)
    persona_emb = embed_persona_text(persona_text, model=model_st)
    pathpool, torch_device = load_pathpool(weights_path, device=device)
    out = predict(
        seg_emb,
        persona_emb,
        pathpool,
        torch_device,
        sharpness=sharpness,
        mismatch_neutral_gain=mismatch_neutral_gain,
        mismatch_emotion_gain=mismatch_emotion_gain,
    )
    out["segment_embeddings"] = seg_emb
    out["persona_embedding"] = persona_emb
    return out


def format_predictions(result: dict) -> str:
    """Human-readable reactions, emotions, and segment weights."""
    lines = []
    if result.get("sharpened"):
        aff = result.get("affinity")
        s = result.get("affinity_s")
        extra = ""
        if aff is not None and s is not None:
            extra = f"  (affinity={aff:+.3f}, s={s:+.2f})"
        lines.append(f"Sharpened predictions{extra}:")
    lines.append("Reactions:")
    for k in REACTION_KEYS:
        v = result["reactions"][k]
        lines.append(f"  {k:14s} {v:6.1%}  {'#' * int(v * 40)}")
    lines.append("Emotions:")
    for k in EMOTION_KEYS:
        v = result["emotions"][k]
        lines.append(f"  {k:14s} {v:6.1%}  {'#' * int(v * 40)}")
    if "segment_weights" in result:
        lines.append("Segment attention:")
        for i, w in enumerate(result["segment_weights"], 1):
            lines.append(f"  segment {i}: {w:.1%}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    from embed_nemotron_rows import row_to_text

    ap = argparse.ArgumentParser(
        description="Predict reactions/emotions for a transcript + persona."
    )
    ap.add_argument(
        "--transcript-jsonl",
        default=str(TRAINING_JSONL),
        help="JSONL file; uses first record's segments by default.",
    )
    ap.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="Which transcript row to use from the JSONL.",
    )
    ap.add_argument(
        "--persona-csv",
        default=str(PERSONA_CSV),
        help="Nemotron CSV; uses first row by default.",
    )
    ap.add_argument(
        "--persona-index",
        type=int,
        default=0,
        help="Which persona row to use from the CSV.",
    )
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--sharpness",
        type=float,
        default=DEFAULT_SHARPNESS,
        help="Amplify persona-specific logit delta (1.0 = raw model output).",
    )
    ap.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Print raw model probabilities (equivalent to --sharpness 1).",
    )
    ap.add_argument(
        "--mismatch-neutral-gain",
        type=float,
        default=1.2,
        help="When standardized affinity s<0, boost neutral and trim like/dislike.",
    )
    ap.add_argument(
        "--mismatch-emotion-gain",
        type=float,
        default=0.8,
        help="When s<0, reduce empathy/relation/joy logits.",
    )
    args = ap.parse_args()

    import json

    import pandas as pd

    with open(args.transcript_jsonl, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == args.record_index:
                record = json.loads(line)
                break
        else:
            raise SystemExit(f"record {args.record_index} not found")

    segments = [s["text"].strip() for s in record["segments"]]
    persona_row = pd.read_csv(args.persona_csv).iloc[args.persona_index]
    persona_text = row_to_text(
        persona_row, [c for c in persona_row.index if c != "uuid"]
    )

    print(f"Transcript id: {record.get('id', args.record_index)}")
    for i, seg in enumerate(segments, 1):
        preview = seg if len(seg) <= 90 else seg[:87] + "..."
        print(f"  [{i}] {preview}")
    print(f"\nPersona row {args.persona_index}:")
    print(f"  {persona_text[:180].replace(chr(10), ' | ')}...\n")

    sharpness = 1.0 if args.no_sharpen else args.sharpness
    result = run_pipeline(
        segments,
        persona_text,
        weights_path=args.weights,
        device=args.device,
        sharpness=sharpness,
        mismatch_neutral_gain=args.mismatch_neutral_gain,
        mismatch_emotion_gain=args.mismatch_emotion_gain,
    )
    print(format_predictions(result))
