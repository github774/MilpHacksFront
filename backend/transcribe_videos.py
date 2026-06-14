#!/usr/bin/env python3
"""Transcribe video files with faster-whisper.

Install dependencies:
    pip install faster-whisper

Requires ffmpeg on PATH.

Example:
    python3 transcribe_videos.py video.mp4
    python3 transcribe_videos.py clips/ --output-dir transcripts/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

from paths import TRANSCRIPTS_DIR

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe input videos with Whisper.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video file(s) or directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TRANSCRIPTS_DIR),
        help=f"Directory for transcript files. Default: {TRANSCRIPTS_DIR}",
    )
    parser.add_argument(
        "--model",
        default="small",
        help='Whisper model size. Default: "small".',
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device. Default: "cpu".',
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help='Compute type. Default: "int8".',
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, e.g. en.",
    )
    return parser.parse_args()


def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for item in inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        if path.is_dir():
            videos.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
                )
            )
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
        else:
            raise ValueError(f"Unsupported input type: {path}")

    if not videos:
        raise ValueError("No video files found.")

    return videos


def extract_audio(video_path: Path, audio_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(audio_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr.strip()}"
        )


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def write_outputs(
    output_stem: Path,
    segments: list[dict[str, object]],
    info: object,
) -> None:
    text = " ".join(segment["text"].strip() for segment in segments).strip()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")

    srt_lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        start = format_timestamp(float(segment["start"]))
        end = format_timestamp(float(segment["end"]))
        srt_lines.extend(
            [
                str(index),
                f"{start} --> {end}",
                str(segment["text"]).strip(),
                "",
            ]
        )
    output_stem.with_suffix(".srt").write_text("\n".join(srt_lines), encoding="utf-8")

    payload = {
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "text": text,
        "segments": segments,
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def transcribe_video(
    model: WhisperModel,
    video_path: Path,
    output_dir: Path,
    language: str | None,
) -> Path:
    output_stem = output_dir / video_path.stem

    with tempfile.TemporaryDirectory(prefix="whisper-audio-") as temp_dir:
        audio_path = Path(temp_dir) / f"{video_path.stem}.wav"
        print(f"Extracting audio: {video_path}")
        extract_audio(video_path, audio_path)

        print(f"Transcribing: {video_path}")
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
        )

        segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in segments_iter
        ]

    write_outputs(output_stem, segments, info)
    print(f"Wrote {output_stem}.txt, .srt, .json")
    return output_stem


def main() -> int:
    args = parse_args()
    videos = collect_videos(args.inputs)
    output_dir = Path(args.output_dir)

    print(
        f"Loading model={args.model!r} device={args.device!r} "
        f"compute_type={args.compute_type!r}"
    )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    for video_path in videos:
        transcribe_video(model, video_path, output_dir, args.language)

    print(f"Done. Transcribed {len(videos)} video(s) to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
