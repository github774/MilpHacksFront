#!/usr/bin/env python3
"""End-to-end tests: transcript + persona text -> embeddings -> persona_pathpool.pt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from common import EMB_DIM, EMOTION_KEYS, REACTION_KEYS
from embed_nemotron_rows import row_to_text
from paths import (
    DEFAULT_EMBED_MODEL,
    PERSONA_CSV,
    PERSONA_WEIGHTS,
    TRAINING_EMBEDDED_JSONL,
)
from pipeline import (
    embed_persona_text,
    embed_transcript_segments,
    format_predictions,
    load_pathpool,
    predict,
    run_pipeline,
)

EMBEDDED_JSONL = TRAINING_EMBEDDED_JSONL
NEMOTRON_CSV = PERSONA_CSV
EMBED_MODEL = DEFAULT_EMBED_MODEL
WEIGHTS = PERSONA_WEIGHTS

SAMPLE_TRANSCRIPT_SEGMENTS = [
    "I almost cried the first time Mapleio actually worked for me. "
    "It turned a chore I dreaded into part of my routine I love.",
    "Real people built this for developers and writers, and you can feel it. "
    "The reviews aren't hype, they're people whose lives got easier.",
    "Try Mapleio today and see for yourself.",
]

SAMPLE_PERSONA_TEXT = (
    "professional_persona: A front-line food service specialist with strong "
    "customer service skills and a routine-driven work ethic.\n"
    "persona: Disciplined, practical, values consistency and community.\n"
    "hobbies_and_interests: Running, home cooking, bullet journaling."
)


def _load_jsonl_record(path: Path, index: int = 0) -> dict:
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == index:
                return json.loads(line)
    raise IndexError(f"record {index} not found in {path}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


@pytest.fixture(scope="module")
def embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@pytest.fixture(scope="module")
def pathpool_model():
    if not WEIGHTS.is_file():
        pytest.skip(f"missing weights: {WEIGHTS}")
    model, device = load_pathpool(WEIGHTS)
    return model, device


@pytest.fixture(scope="module")
def nemotron_persona_text() -> str:
    import pandas as pd

    row = pd.read_csv(NEMOTRON_CSV, nrows=1).iloc[0]
    text_columns = [c for c in row.index if c != "uuid"]
    return row_to_text(row, text_columns)


class TestEmbeddings:
    def test_transcript_segments_shape(self, embed_model):
        emb = embed_transcript_segments(SAMPLE_TRANSCRIPT_SEGMENTS, model=embed_model)
        assert emb.shape == (len(SAMPLE_TRANSCRIPT_SEGMENTS), EMB_DIM)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-4, atol=1e-4)

    def test_persona_embedding_shape(self, embed_model):
        emb = embed_persona_text(SAMPLE_PERSONA_TEXT, model=embed_model)
        assert emb.shape == (EMB_DIM,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    @pytest.mark.skipif(
        not EMBEDDED_JSONL.is_file(),
        reason="training_data_with_embeddings.jsonl not present",
    )
    def test_segment_embeddings_match_reference(self, embed_model):
        """Fresh bge-small encodes should match stored training embeddings."""
        ref = _load_jsonl_record(EMBEDDED_JSONL, 0)
        texts = [s["text"].strip() for s in ref["segments"]]
        ref_emb = np.array([s["embedding"] for s in ref["segments"]], dtype=np.float32)
        fresh = embed_transcript_segments(texts, model=embed_model)

        for i in range(len(texts)):
            cos = _cosine(fresh[i], ref_emb[i])
            assert cos > 0.995, f"segment {i} cosine={cos:.4f}"


class TestPrediction:
    def test_predict_output_schema(self, embed_model, pathpool_model):
        model, device = pathpool_model
        seg = embed_transcript_segments(SAMPLE_TRANSCRIPT_SEGMENTS, model=embed_model)
        persona = embed_persona_text(SAMPLE_PERSONA_TEXT, model=embed_model)
        out = predict(seg, persona, model, device)

        assert set(out["reactions"]) == set(REACTION_KEYS)
        assert set(out["emotions"]) == set(EMOTION_KEYS)
        assert len(out["segment_weights"]) == len(SAMPLE_TRANSCRIPT_SEGMENTS)

        react = np.array([out["reactions"][k] for k in REACTION_KEYS])
        emo = np.array([out["emotions"][k] for k in EMOTION_KEYS])
        np.testing.assert_allclose(react.sum(), 1.0, rtol=1e-4, atol=1e-4)
        assert np.all(react >= 0) and np.all(react <= 1)
        assert np.all(emo >= 0) and np.all(emo <= 1)
        assert abs(sum(out["segment_weights"]) - 1.0) < 1e-5

    def test_persona_changes_prediction(self, embed_model, pathpool_model):
        """Different personas should yield different reaction distributions."""
        model, device = pathpool_model
        seg = embed_transcript_segments(SAMPLE_TRANSCRIPT_SEGMENTS, model=embed_model)

        persona_a = embed_persona_text(SAMPLE_PERSONA_TEXT, model=embed_model)
        persona_b = embed_persona_text(
            "professional_persona: A skeptical cybersecurity analyst who distrusts "
            "marketing hype and prefers rigorous evidence.\n"
            "persona: Cautious, analytical, allergic to upsells.",
            model=embed_model,
        )
        out_a = predict(seg, persona_a, model, device)
        out_b = predict(seg, persona_b, model, device)

        react_a = np.array([out_a["reactions"][k] for k in REACTION_KEYS])
        react_b = np.array([out_b["reactions"][k] for k in REACTION_KEYS])
        assert not np.allclose(react_a, react_b, atol=1e-3)

    def test_shuffled_persona_differs_from_correct(self, embed_model, pathpool_model):
        model, device = pathpool_model
        seg = embed_transcript_segments(SAMPLE_TRANSCRIPT_SEGMENTS, model=embed_model)
        persona = embed_persona_text(SAMPLE_PERSONA_TEXT, model=embed_model)
        wrong = embed_persona_text(
            "persona: Completely unrelated random interests in medieval pottery.",
            model=embed_model,
        )
        correct = predict(seg, persona, model, device)
        mismatched = predict(seg, wrong, model, device)
        r0 = np.array([correct["reactions"][k] for k in REACTION_KEYS])
        r1 = np.array([mismatched["reactions"][k] for k in REACTION_KEYS])
        assert not np.allclose(r0, r1, atol=1e-3)


class TestFullPipeline:
    def test_run_pipeline_end_to_end(self, embed_model):
        if not WEIGHTS.is_file():
            pytest.skip(f"missing weights: {WEIGHTS}")

        out = run_pipeline(
            SAMPLE_TRANSCRIPT_SEGMENTS,
            SAMPLE_PERSONA_TEXT,
            embed_model=embed_model,
            sharpness=1.0,
        )
        print("\n" + format_predictions(out))
        assert out["segment_embeddings"].shape == (3, EMB_DIM)
        assert out["persona_embedding"].shape == (EMB_DIM,)
        assert "like" in out["reactions"]
        assert "joy" in out["emotions"]

    @pytest.mark.skipif(
        not NEMOTRON_CSV.is_file(),
        reason="nemotron_sample_10k.csv not present",
    )
    def test_pipeline_with_nemotron_persona_row(
        self, embed_model, nemotron_persona_text
    ):
        if not WEIGHTS.is_file():
            pytest.skip(f"missing weights: {WEIGHTS}")

        out = run_pipeline(
            SAMPLE_TRANSCRIPT_SEGMENTS,
            nemotron_persona_text,
            embed_model=embed_model,
            sharpness=1.0,
        )
        assert out["reactions"]["neutral"] >= 0
        assert out["emotions"]["curiosity"] <= 1

    def test_pipeline_is_deterministic(self, embed_model):
        if not WEIGHTS.is_file():
            pytest.skip(f"missing weights: {WEIGHTS}")

        kwargs = dict(
            segment_texts=SAMPLE_TRANSCRIPT_SEGMENTS,
            persona_text=SAMPLE_PERSONA_TEXT,
            embed_model=embed_model,
            sharpness=1.0,
        )
        a = run_pipeline(**kwargs)
        b = run_pipeline(**kwargs)
        for key in REACTION_KEYS:
            assert a["reactions"][key] == pytest.approx(b["reactions"][key])
        for key in EMOTION_KEYS:
            assert a["emotions"][key] == pytest.approx(b["emotions"][key])


if __name__ == "__main__":
    out = run_pipeline(SAMPLE_TRANSCRIPT_SEGMENTS, SAMPLE_PERSONA_TEXT)
    print(format_predictions(out))
