#!/usr/bin/env python3
"""Embed each segment of each transcript JSONL record.

Install dependencies:
    python3 -m pip install sentence-transformers torch

Example:
    python3 embed_transcript_segments.py \\
        --input training_data.jsonl \\
        --output training_data_with_embeddings.jsonl
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from sentence_transformers import SentenceTransformer


from paths import (
    DEFAULT_CSV_OUTPUT,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
)
DEFAULT_BATCH_SIZE = 32
DEFAULT_PRECISION = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one text embedding per transcript segment and store results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL with an embedding field on each segment.",
    )
    parser.add_argument(
        "--csv-output",
        default=DEFAULT_CSV_OUTPUT,
        help="Flat CSV with one row per segment (id, segment_index, embedding, ...).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformers model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Model encode batch size.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places to keep in serialized embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, e.g. 'mps', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the flat segment embeddings CSV.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "segments" not in record:
                raise ValueError(f"Line {line_number} missing 'segments'.")
            records.append(record)
    return records


def segment_text(segment: dict) -> str:
    return str(segment.get("text", "")).strip()


def serialize_embedding(embedding, precision: int) -> list[float]:
    return [round(float(value), precision) for value in embedding]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    csv_output_path = Path(args.csv_output)

    records = load_records(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    segment_refs: list[tuple[int, int]] = []
    texts: list[str] = []
    for record_index, record in enumerate(records):
        for segment_index, segment in enumerate(record["segments"]):
            text = segment_text(segment)
            if not text:
                raise ValueError(
                    f"Record {record.get('id', record_index)} segment {segment_index} is empty."
                )
            segment_refs.append((record_index, segment_index))
            texts.append(text)

    print(f"Loaded {len(records)} transcripts ({len(texts)} segments) from {input_path}")
    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    enriched_records = [deepcopy(record) for record in records]
    csv_rows: list[dict] = []

    for (record_index, segment_index), embedding in zip(segment_refs, embeddings):
        serialized = serialize_embedding(embedding, args.precision)
        enriched_records[record_index]["segments"][segment_index]["embedding"] = serialized

        record = enriched_records[record_index]
        segment = record["segments"][segment_index]
        csv_rows.append(
            {
                "id": record.get("id", ""),
                "segment_index": segment_index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment_text(segment),
                "embedding": json.dumps(serialized, separators=(",", ":")),
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for record in enriched_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(enriched_records)} records to {output_path}")

    if not args.no_csv:
        import pandas as pd

        pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False)
        print(f"Wrote {len(csv_rows)} segment rows to {csv_output_path}")


if __name__ == "__main__":
    main()
