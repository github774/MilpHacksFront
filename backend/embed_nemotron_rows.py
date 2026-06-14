#!/usr/bin/env python3
"""Embed Nemotron persona rows into one vector per UUID.

Install dependencies:
    python3 -m pip install pandas sentence-transformers torch

Example:
    python3 embed_nemotron_rows.py \
        --input nemotron_sample_10k.csv \
        --output nemotron_sample_10k_embeddings_noname.csv \
        --model BAAI/bge-small-en-v1.5

Weighted field averaging (legacy):
    python3 embed_nemotron_rows.py \
        --input nemotron_sample_10k.csv \
        --output nemotron_sample_10k_embeddings_noname_weighted.csv \
        --weighted
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


from paths import (
    DEFAULT_EMBED_MODEL,
    PERSONA_CSV,
    PERSONA_EMBEDDINGS_CSV,
    PERSONA_EMBEDDINGS_MINMAX_CSV,
    PERSONA_EMBEDDINGS_WEIGHTED_CSV,
)

DEFAULT_INPUT = PERSONA_CSV
DEFAULT_OUTPUT = PERSONA_EMBEDDINGS_CSV
DEFAULT_BATCH_SIZE = 16
DEFAULT_CHUNK_SIZE = 512
DEFAULT_PRECISION = 6

TEXT_COLUMN_WEIGHTS = {
    "professional_persona": 3.0,
    "skills_and_expertise": 2.5,
    "skills_and_expertise_list": 2.0,
    "career_goals_and_ambitions": 2.0,
    "persona": 2.0,
    "hobbies_and_interests": 1.5,
    "hobbies_and_interests_list": 1.25,
    "cultural_background": 1.25,
    "sports_persona": 1.0,
    "arts_persona": 1.0,
    "travel_persona": 1.0,
    "culinary_persona": 1.0,
}
DEFAULT_COLUMN_WEIGHT = 0.75
START_NAME_RE = re.compile(
    r"^([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,3})"
    r"(?=\s+(?:is|was|has|finds|fuels|prefers|showcases|blends|combines|spends|enjoys)\b|,)"
)
NAME_STOPWORDS = {
    "A",
    "An",
    "The",
    "In",
    "Outside",
    "She",
    "He",
    "They",
    "Their",
    "Her",
    "His",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one text embedding per Nemotron row and save uuid,embedding CSV."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"SentenceTransformers model name. Default: {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Model encode batch size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CSV rows to process per chunk.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places to keep in the JSON embedding stored in CSV.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, e.g. 'mps', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Average per-field embeddings with column weights instead of one row embedding.",
    )
    return parser.parse_args()


def column_weight(column: str) -> float:
    return TEXT_COLUMN_WEIGHTS.get(column, DEFAULT_COLUMN_WEIGHT)


def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def clean_text_value(value) -> str:
    if is_missing(value):
        return ""

    return str(value).strip()


def is_plausible_name(candidate: str) -> bool:
    first_word = candidate.split()[0]
    return first_word not in NAME_STOPWORDS


def extract_person_names(row: pd.Series, columns: Iterable[str]) -> list[str]:
    """Infer generated person names so they do not dominate similarity."""
    priority_columns = [
        "professional_persona",
        "persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
    ]
    ordered_columns = [
        column for column in priority_columns if column in columns
    ] + [column for column in columns if column not in priority_columns]

    for column in ordered_columns:
        text = clean_text_value(row.get(column))
        if not text:
            continue

        start_match = START_NAME_RE.search(text)
        if start_match and is_plausible_name(start_match.group(1)):
            name = start_match.group(1)
            names = {name, name.split()[0]}
            # Remove longer names first so "Mary Alberti" is stripped before "Mary".
            return sorted(names, key=len, reverse=True)

    return []


def strip_names(text: str, names: Iterable[str]) -> str:
    stripped = text
    for name in names:
        escaped = re.escape(name)
        stripped = re.sub(rf"\b{escaped}'s\b", "the person's", stripped)
        stripped = re.sub(rf"\b{escaped}\b", "the person", stripped)

    return re.sub(r"\s+", " ", stripped).strip()


def row_to_text(row: pd.Series, columns: Iterable[str]) -> str:
    """Convert one row into a single labeled text block with names removed."""
    names = extract_person_names(row, columns)
    parts: list[str] = []
    for column in columns:
        text = clean_text_value(row.get(column))
        if not text:
            continue

        sanitized = strip_names(text, names)
        parts.append(f"{column}: {sanitized}")

    return "\n".join(parts)


def row_to_weighted_texts(row: pd.Series, columns: Iterable[str]) -> list[tuple[str, float]]:
    """Convert one row into labeled, weighted text blocks for vector averaging."""
    names = extract_person_names(row, columns)
    weighted_texts: list[tuple[str, float]] = []
    for column in columns:
        text = clean_text_value(row.get(column))
        if not text:
            continue

        sanitized = strip_names(text, names)
        weighted_texts.append((f"{column}: {sanitized}", column_weight(column)))

    return weighted_texts


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def encode_rows(
    model: SentenceTransformer,
    chunk: pd.DataFrame,
    text_columns: list[str],
    batch_size: int,
) -> np.ndarray:
    texts = [row_to_text(row, text_columns) for _, row in chunk.iterrows()]
    if not any(texts):
        raise ValueError("No embeddable text found in chunk.")

    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def encode_weighted_rows(
    model: SentenceTransformer,
    chunk: pd.DataFrame,
    text_columns: list[str],
    batch_size: int,
) -> np.ndarray:
    weighted_rows = [
        row_to_weighted_texts(row, text_columns) for _, row in chunk.iterrows()
    ]
    flat_texts = [
        text for weighted_texts in weighted_rows for text, _ in weighted_texts
    ]
    if not flat_texts:
        raise ValueError("No embeddable text found in chunk.")

    flat_embeddings = model.encode(
        flat_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    flat_embeddings = np.asarray(flat_embeddings, dtype=np.float32)

    output = np.zeros((len(weighted_rows), flat_embeddings.shape[1]), dtype=np.float32)
    cursor = 0
    for row_index, weighted_texts in enumerate(weighted_rows):
        for _, weight in weighted_texts:
            output[row_index] += flat_embeddings[cursor] * weight
            cursor += 1

    return normalize_embeddings(output)


def serialize_embedding(embedding, precision: int) -> str:
    rounded = [round(float(value), precision) for value in embedding]
    return json.dumps(rounded, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    model = SentenceTransformer(args.model, device=args.device)
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(input_path, chunksize=args.chunk_size):
        if "uuid" not in chunk.columns:
            raise ValueError("Input CSV must contain a 'uuid' column.")

        text_columns = [column for column in chunk.columns if column != "uuid"]
        encode = encode_weighted_rows if args.weighted else encode_rows
        embeddings = encode(
            model,
            chunk,
            text_columns,
            args.batch_size,
        )

        output = pd.DataFrame(
            {
                "uuid": chunk["uuid"].astype(str).to_list(),
                "embedding": [
                    serialize_embedding(embedding, args.precision)
                    for embedding in embeddings
                ],
            }
        )
        output.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )

        total_rows += len(output)
        first_chunk = False
        print(f"Wrote {total_rows} embeddings to {output_path}")


if __name__ == "__main__":
    main()
