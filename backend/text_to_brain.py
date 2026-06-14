#!/usr/bin/env python3
"""Standalone text -> brain coordinates pipeline (TribeV2).

Takes a text stimulus and emits, for every predicted fMRI timestep, the most
active cortical locations as real 3D coordinates on the fsaverage5 surface.

The model path mirrors the official ``tribe_demo.ipynb`` text section:
text is converted to speech (gTTS), transcribed back to word-level events,
then run through ``facebook/tribev2`` to produce per-TR activations of shape
``(n_timesteps, n_vertices)``. Each vertex index is mapped to its (x, y, z)
position on the fsaverage5 mesh (left hemisphere first, then right).

Usage:
    python text_to_brain.py "To be or not to be, that is the question."
    python text_to_brain.py --text-file passage.txt --out brain.json
    echo "some text" | python text_to_brain.py --top-k 64 --out brain.json

First run downloads ~1-3 GB of model weights into ``HF_HOME``. GPU strongly
recommended; CPU works but is slow.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Use the vendored tribev2-main source tree in preference to any installed
# copy, so this script runs against the local package regardless of which
# interpreter / virtualenv invokes it.
_TRIBEV2_SRC = Path(__file__).resolve().parent / "tribev2-main"
if _TRIBEV2_SRC.is_dir() and str(_TRIBEV2_SRC) not in sys.path:
    sys.path.insert(0, str(_TRIBEV2_SRC))

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "./hf_cache"))

import torch  # noqa: E402

torch.set_float32_matmul_precision("high")
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

warnings.filterwarnings(
    "ignore",
    message=".*event_types has not been set.*",
    category=UserWarning,
    module="neuralset.extractors.base",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="x_transformers.x_transformers",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("text-to-brain")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _patch_whisperx_for_cuda_cpu() -> None:
    """Tribe's whisperx wrapper assumes cuda+float16; fall back to cpu+float32.

    The text pipeline still runs whisperx (text -> gTTS audio -> transcription),
    so this patch is required for CPU-only boxes.
    """
    import tribev2.eventstransforms as et
    from tribev2.eventstransforms import logger as _et_logger

    language_codes = dict(
        english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
    )

    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "float32"

        wav_filename = Path(wav_filename)
        with tempfile.TemporaryDirectory() as output_dir:
            _et_logger.info("Running whisperx (%s, %s)...", device, compute_type)
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model", "large-v3",
                "--language", language_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", "16",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            cmd = [c for c in cmd if c]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")
            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                words.append({
                    "text": word["word"].replace('"', ""),
                    "start": word["start"],
                    "duration": word["end"] - word["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                })
        return pd.DataFrame(words)

    et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _get_transcript_from_audio
    )


def _fsaverage5_coords(n_vertices: int) -> np.ndarray:
    """Return (n_vertices, 3) fsaverage5 surface coords, left hemi then right.

    Vertex order matches TribeV2's prediction layout: lh vertices [0, n/2),
    then rh vertices [n/2, n).
    """
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    left = surface.load_surf_mesh(fs["pial_left"])[0]
    right = surface.load_surf_mesh(fs["pial_right"])[0]
    coords = np.vstack([left, right]).astype(np.float32)
    if coords.shape[0] != n_vertices:
        logger.warning(
            "fsaverage5 vertex count (%d) != prediction vertices (%d); "
            "coordinates may be misaligned.",
            coords.shape[0], n_vertices,
        )
    return coords


def _destrieux_region_lookup(n_vertices: int):
    """Return a fn mapping a global vertex index to its Destrieux parcel name."""
    from nilearn import datasets

    try:
        atlas = datasets.fetch_atlas_surf_destrieux()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Destrieux atlas unavailable (%r); regions omitted.", exc)
        return lambda gv: None

    lut = atlas["lut"]
    id_to_name = {int(r["index"]): str(r["name"]) for _, r in lut.iterrows()}
    map_left = np.asarray(atlas["map_left"])
    map_right = np.asarray(atlas["map_right"])
    n_hemi = n_vertices // 2

    def lookup(global_vertex: int):
        gv = int(global_vertex)
        if gv < 0 or gv >= n_vertices:
            return None
        if gv < n_hemi:
            rid = int(map_left[gv]) if gv < map_left.shape[0] else -1
        else:
            local = gv - n_hemi
            rid = int(map_right[local]) if local < map_right.shape[0] else -1
        return id_to_name.get(rid)

    return lookup


def load_model():
    from tribev2.demo_utils import TribeModel

    cache_folder = Path(os.environ.get("TRIBE_CACHE", "./cache"))
    cache_folder.mkdir(parents=True, exist_ok=True)
    dev = _device()
    logger.info("Loading facebook/tribev2 (device=%s)...", dev)
    return TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=cache_folder,
        device=dev,
        config_update={
            "data.subject_id.event_types": ("Word", "Audio", "Video", "Image"),
            "data.text_feature.device": dev,
            "data.audio_feature.device": dev,
            "data.image_feature.image.device": dev,
        },
    )


def text_to_brain(text: str, top_k: int = 64) -> dict[str, Any]:
    """Run TribeV2 on text and return per-timestep brain coordinate payload."""
    if not text.strip():
        raise ValueError("Input text is empty.")

    _patch_whisperx_for_cuda_cpu()
    model = load_model()
    model.remove_empty_segments = False

    cache_folder = Path(os.environ.get("TRIBE_CACHE", "./cache"))
    text_path = cache_folder / "text_to_brain_input.txt"
    text_path.write_text(text, encoding="utf-8")

    logger.info("Building events from text (TTS + transcription)...")
    df = model.get_events_dataframe(text_path=str(text_path))

    logger.info("Predicting brain responses...")
    with torch.inference_mode():
        preds, segments = model.predict(events=df)
    preds = np.asarray(preds, dtype=np.float32)
    n_t, n_v = preds.shape
    logger.info("Predictions shape: (%d timesteps, %d vertices)", n_t, n_v)

    coords = _fsaverage5_coords(n_v)
    region_of = _destrieux_region_lookup(n_v)
    n_hemi = n_v // 2
    k = max(1, min(int(top_k), n_v))
    global_max_abs = float(np.max(np.abs(preds))) if preds.size else 1.0
    norm_denom = max(global_max_abs, 1e-8)

    timesteps = []
    for i in range(n_t):
        row = preds[i]
        abs_row = np.abs(row)
        top_idx = np.argsort(-abs_row)[:k]
        seg = segments[i] if i < len(segments) else None
        start = float(seg.start) if seg is not None else float(i)
        duration = float(seg.duration) if seg is not None else 1.0

        points = []
        for vi in top_idx:
            vi = int(vi)
            cx, cy, cz = (
                coords[vi] if vi < coords.shape[0] else (0.0, 0.0, 0.0)
            )
            signed = float(row[vi])
            points.append({
                "vertex": vi,
                "hemisphere": "left" if vi < n_hemi else "right",
                "region": region_of(vi),
                "x": round(float(cx), 4),
                "y": round(float(cy), 4),
                "z": round(float(cz), 4),
                "activation": round(signed, 6),
                "activation_abs_norm_0_to_1": round(
                    float(abs_row[vi] / norm_denom), 6
                ),
            })

        timesteps.append({
            "timestep_index": i,
            "time_start_sec": round(start, 4),
            "time_end_sec": round(start + duration, 4),
            "activation_l2": round(float(np.linalg.norm(row)), 4),
            "points": points,
        })

    return {
        "source": "facebook/tribev2 (text -> TTS -> transcription -> fMRI)",
        "input_text": text,
        "device": _device(),
        "mesh": "fsaverage5",
        "coordinate_space": "fsaverage5 pial surface (mm), left hemi then right",
        "shape_timesteps_vertices": [int(n_t), int(n_v)],
        "top_k_vertices_per_timestep": int(k),
        "global_abs_activation_max": round(global_max_abs, 6),
        "timesteps": timesteps,
    }


def _read_input_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("No text provided. Pass text as an argument, --text-file, or stdin.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text -> brain coordinates per timestep (TribeV2), JSON output."
    )
    parser.add_argument("text", nargs="?", help="Text stimulus (or use --text-file / stdin).")
    parser.add_argument("--text-file", help="Path to a .txt file with the stimulus.")
    parser.add_argument(
        "--top-k", type=int, default=64,
        help="Number of most-active vertices to keep per timestep (default: 64).",
    )
    parser.add_argument(
        "--out", help="Write JSON here instead of stdout.",
    )
    args = parser.parse_args()

    text = _read_input_text(args)
    payload = text_to_brain(text, top_k=args.top_k)
    output = json.dumps(payload, indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        logger.info("Wrote %s", args.out)
    else:
        print(output)


if __name__ == "__main__":
    main()
