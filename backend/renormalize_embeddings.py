#!/usr/bin/env python3
"""Min-max scale embedding columns across an entire embeddings CSV.

Each embedding dimension is scaled independently so that, across all rows:
    min(column) -> 0
    max(column) -> 1

Example:
    python3 renormalize_embeddings.py \
        --input nemotron_sample_10k_embeddings_noname_weighted.csv \
        --output nemotron_sample_10k_embeddings_noname_weighted_minmax.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


from paths import (
    PERSONA_EMBEDDINGS_MINMAX_CSV,
    PERSONA_EMBEDDINGS_WEIGHTED_CSV,
)

DEFAULT_INPUT = PERSONA_EMBEDDINGS_WEIGHTED_CSV
DEFAULT_OUTPUT = PERSONA_EMBEDDINGS_MINMAX_CSV
DEFAULT_PRECISION = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Min-max normalize each embedding dimension across all rows."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input uuid,embedding CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument(
        "--stats-output",
        default=None,
        help="Optional JSON path for per-dimension min/max values.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="Decimal places for serialized embedding values.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    uuids: list[str] = []
    vectors: list[list[float]] = []

    frame = pd.read_csv(path)
    if "uuid" not in frame.columns or "embedding" not in frame.columns:
        raise ValueError("Input CSV must contain uuid and embedding columns.")

    for _, row in frame.iterrows():
        uuids.append(str(row["uuid"]))
        vectors.append(json.loads(row["embedding"]))

    return uuids, np.asarray(vectors, dtype=np.float64)


def minmax_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins
    constant_dims = ranges == 0

    scaled = np.zeros_like(matrix, dtype=np.float64)
    nonzero = ~constant_dims
    scaled[:, nonzero] = (matrix[:, nonzero] - mins[nonzero]) / ranges[nonzero]
    scaled[:, constant_dims] = 0.0

    return scaled, mins, maxs


def serialize_embedding(embedding: np.ndarray, precision: int) -> str:
    rounded = [round(float(value), precision) for value in embedding]
    return json.dumps(rounded, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = (
        Path(args.stats_output)
        if args.stats_output
        else output_path.with_suffix(".minmax.json")
    )

    uuids, matrix = load_embeddings(input_path)
    scaled, mins, maxs = minmax_scale(matrix)

    output = pd.DataFrame(
        {
            "uuid": uuids,
            "embedding": [
                serialize_embedding(row, args.precision) for row in scaled
            ],
        }
    )
    output.to_csv(output_path, index=False)

    stats = {
        "input": str(input_path),
        "rows": len(uuids),
        "dimensions": int(matrix.shape[1]),
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Wrote {len(output)} rows to {output_path}")
    print(f"Saved min/max stats to {stats_path}")
    print(f"Scaled range check: min={scaled.min():.6f}, max={scaled.max():.6f}")


if __name__ == "__main__":
    main()
