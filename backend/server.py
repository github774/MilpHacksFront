#!/usr/bin/env python3
"""FastAPI server for the SwarmMind content-simulation product.

Endpoints
---------
GET  /api/health      -> service + model + ollama status
POST /api/transcribe  -> multipart video upload -> Whisper transcript + segments
POST /api/simulate    -> transcript/text + n_seeds -> full network-simulation JSON
POST /api/chat        -> stream a persona roleplay reply from Ollama (qwen2.5:0.5b; set OLLAMA_MODEL for larger)

Run it with the interpreter that has the project deps:

    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
        -m uvicorn server:app --host 0.0.0.0 --port 8000

Heavy resources (persona catalog, embedding model, path-pool weights, Whisper
model) are loaded once and reused across requests.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import network_simulation as ns
from network_simulation import PersonaCatalog, run_simulation
from paths import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_SHARPNESS,
    PERSONA_CSV,
    PERSONA_EMBEDDINGS_CSV,
    PERSONA_WEIGHTS,
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg", ".wmv",
}

app = FastAPI(title="SwarmMind API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Lazy singletons
# --------------------------------------------------------------------------- #
class _Models:
    """Holds the heavy ML resources, loaded on first use."""

    def __init__(self) -> None:
        self.catalog: PersonaCatalog | None = None
        self.embed_model = None
        self.pathpool = None
        self.torch_device = None
        self.whisper = None
        self._sim_lock = threading.Lock()
        self._whisper_lock = threading.Lock()
        self._load_lock = threading.Lock()

    # -- simulation stack -- #
    def ensure_simulation(self) -> None:
        if self.catalog is not None:
            return
        with self._load_lock:
            if self.catalog is not None:
                return
            from sentence_transformers import SentenceTransformer

            from pipeline import load_pathpool

            print("[models] loading persona catalog ...", flush=True)
            catalog = PersonaCatalog.load(PERSONA_CSV, PERSONA_EMBEDDINGS_CSV)
            # Sanitize any NaN/inf rows so cosine-similarity stays well-defined.
            emb = np.nan_to_num(catalog.embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            catalog.embeddings = (emb / np.maximum(norms, 1e-9)).astype(np.float32)
            print(f"[models] catalog loaded: {len(catalog)} personas", flush=True)

            print("[models] loading embedding model ...", flush=True)
            embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL)

            print("[models] loading path-pool weights ...", flush=True)
            pathpool, torch_device = load_pathpool(PERSONA_WEIGHTS)

            # Patch network_simulation so run_simulation reuses these singletons
            # instead of reloading them on every request.
            ns.SentenceTransformer = lambda *a, **k: embed_model  # type: ignore
            ns.load_pathpool = lambda *a, **k: (pathpool, torch_device)  # type: ignore

            self.catalog = catalog
            self.embed_model = embed_model
            self.pathpool = pathpool
            self.torch_device = torch_device
            print("[models] simulation stack ready", flush=True)

    # -- whisper -- #
    def ensure_whisper(self):
        if self.whisper is not None:
            return self.whisper
        with self._load_lock:
            if self.whisper is None:
                from faster_whisper import WhisperModel

                print(f"[models] loading whisper '{WHISPER_MODEL_SIZE}' ...", flush=True)
                self.whisper = WhisperModel(
                    WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
                )
                print("[models] whisper ready", flush=True)
        return self.whisper


MODELS = _Models()


@app.on_event("startup")
def _prewarm() -> None:
    """Warm the simulation stack in the background so first request is instant."""
    threading.Thread(target=MODELS.ensure_simulation, daemon=True).start()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def split_into_segments(text: str) -> list[str]:
    """Split free text into sentence-ish segments for the model."""
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    buf = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # Merge tiny fragments so each segment carries enough signal.
        if len(buf) + len(piece) < 40 and buf:
            buf = f"{buf} {piece}"
        else:
            if buf:
                segments.append(buf)
            buf = piece
    if buf:
        segments.append(buf)
    return segments or [text]


def extract_audio(video_path: Path, audio_path: Path) -> None:
    import subprocess

    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")


def _segment_texts(segments: list[Any] | None) -> list[str]:
    out: list[str] = []
    for seg in segments or []:
        if isinstance(seg, str):
            text = seg.strip()
        elif isinstance(seg, dict):
            text = str(seg.get("text", "")).strip()
        else:
            text = str(seg).strip()
        if text:
            out.append(text)
    return out


def resolve_chat_transcript(
    content_text: str | None,
    content_segments: list[dict[str, Any]] | None,
    content_meta: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve full transcript text + segment list from chat request fields."""
    meta = content_meta or {}
    raw_segments: list[Any] = list(content_segments or meta.get("segments") or [])

    if not raw_segments:
        whisper = meta.get("whisper_segments")
        if isinstance(whisper, list):
            raw_segments = whisper

    normalized: list[dict[str, Any]] = []
    for i, seg in enumerate(raw_segments):
        if isinstance(seg, str):
            text = seg.strip()
            if text:
                normalized.append({"text": text, "index": i})
        elif isinstance(seg, dict):
            normalized.append(seg)

    text = (content_text or meta.get("text") or "").strip()
    if not text:
        text = " ".join(_segment_texts(normalized)).strip()

    return text, normalized


def persona_system_prompt(
    record: dict[str, Any],
    archetype: str,
    occupation: str,
    reaction: dict[str, Any] | None,
    content_text: str | None,
    content_segments: list[dict[str, Any]] | None = None,
    content_meta: dict[str, Any] | None = None,
    simulation: dict[str, Any] | None = None,
) -> str:
    """Build a vivid roleplay system prompt with full content + simulation context."""

    def g(key: str) -> str:
        v = record.get(key)
        if v is None:
            return ""
        s = str(v).strip()
        return "" if s.lower() in {"nan", "none", "unknown"} else s

    bio_bits: list[str] = []
    age, sex = g("age"), g("sex")
    if age:
        bio_bits.append(f"{age}-year-old")
    if sex:
        bio_bits.append(sex.lower())
    if occupation and occupation != "unknown":
        bio_bits.append(occupation.replace("_", " "))
    headline = " ".join(bio_bits).strip() or archetype

    locale = ", ".join(b for b in [g("city"), g("state"), g("country")] if b)

    detail_fields = [
        ("Background", "professional_persona"),
        ("About", "persona"),
        ("Culture", "cultural_background"),
        ("Skills", "skills_and_expertise"),
        ("Skills (list)", "skills_and_expertise_list"),
        ("Hobbies", "hobbies_and_interests"),
        ("Hobbies (list)", "hobbies_and_interests_list"),
        ("Goals", "career_goals_and_ambitions"),
        ("Education", "education_level"),
        ("Field", "bachelors_field"),
        ("Family", "marital_status"),
        ("Sports", "sports_persona"),
        ("Arts", "arts_persona"),
        ("Travel", "travel_persona"),
        ("Culinary", "culinary_persona"),
    ]
    details: list[str] = []
    for label, key in detail_fields:
        val = g(key)
        if val:
            details.append(f"- {label}: {val[:500]}")

    lines = [
        f"You are roleplaying as a real person: a {headline}"
        + (f" from {locale}" if locale else "")
        + f" ({archetype}).",
        "",
        "=== YOUR PROFILE (stay in character) ===",
        *details,
    ]

    meta = content_meta or {}
    segments = list(content_segments or meta.get("segments") or [])
    full_text = (content_text or meta.get("text") or "").strip()
    if not full_text:
        full_text = " ".join(_segment_texts(segments)).strip()
    if not segments and meta.get("whisper_segments"):
        whisper = meta.get("whisper_segments")
        if isinstance(whisper, list):
            segments = [s for s in whisper if isinstance(s, dict)]
            if not full_text:
                full_text = " ".join(_segment_texts(segments)).strip()

    if full_text or segments:
        lines += ["", "=== CONTENT YOU JUST SAW IN YOUR FEED ==="]
        source = meta.get("source") or meta.get("input_mode")
        if source == "video" or meta.get("video_filename") or meta.get("video"):
            fname = meta.get("video_filename") or meta.get("video") or "uploaded video"
            lines.append(f"Format: short-form video — \"{fname}\"")
            if meta.get("transcribe_seconds") is not None:
                lines.append(
                    f"Audio transcribed locally (Whisper, {meta.get('transcribe_seconds')}s processing)."
                )
        elif source == "text":
            lines.append("Format: text post / caption pasted into the feed.")
        else:
            lines.append(f"Format: {source or 'social post'}.")

        if meta.get("language"):
            lang = str(meta["language"]).upper()
            prob = meta.get("language_probability")
            if prob is not None:
                lines.append(f"Language: {lang} ({float(prob) * 100:.0f}% confidence)")
            else:
                lines.append(f"Language: {lang}")
        if meta.get("duration") is not None:
            lines.append(f"Clip length: {float(meta['duration']):.1f} seconds")

        if full_text:
            lines += ["", "Full transcript / caption (verbatim):", full_text]

        if segments:
            lines.append("")
            lines.append("Segment breakdown (what the model parsed):")
            for i, seg in enumerate(segments):
                if isinstance(seg, str):
                    lines.append(f"  {i + 1}. {seg.strip()}")
                    continue
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                start, end = seg.get("start"), seg.get("end")
                if start is not None and end is not None:
                    lines.append(f"  {i + 1}. [{float(start):.1f}s–{float(end):.1f}s] {text}")
                else:
                    lines.append(f"  {i + 1}. {text}")

        run_summary = meta.get("simulation_summary")
        if isinstance(run_summary, dict) and run_summary:
            lines += ["", "=== HOW THIS CONTENT SPREAD IN THE SIMULATION (background context) ==="]
            lines.append(
                f"Total people exposed: {run_summary.get('total_exposed')} "
                f"across {run_summary.get('waves_completed')} share waves."
            )
            lines.append(
                f"Reactions: {run_summary.get('likes')} likes, "
                f"{run_summary.get('dislikes')} dislikes, "
                f"{run_summary.get('like_shares')} like-shares, "
                f"{run_summary.get('dislike_shares')} dislike-shares, "
                f"{run_summary.get('neutral')} neutral scroll-pasts."
            )
            if run_summary.get("virality_score") is not None:
                lines.append(f"Virality score: {float(run_summary['virality_score']):.2f}/100")

        liked = meta.get("top_liked_archetypes")
        disliked = meta.get("top_disliked_archetypes")
        if liked or disliked:
            lines.append("")
            if liked:
                parts = [
                    f"{a.get('archetype', a)} ({a.get('count', '?')})"
                    if isinstance(a, dict)
                    else str(a)
                    for a in liked[:3]
                ]
                lines.append(f"Archetypes that liked it most: {', '.join(parts)}")
            if disliked:
                parts = [
                    f"{a.get('archetype', a)} ({a.get('count', '?')})"
                    if isinstance(a, dict)
                    else str(a)
                    for a in disliked[:3]
                ]
                lines.append(f"Archetypes that disliked it most: {', '.join(parts)}")

        attn = meta.get("segment_attention_mean")
        if attn and segments and len(attn) == len(segments):
            lines.append("")
            lines.append("Average audience attention per segment (simulation-wide):")
            for i, seg in enumerate(segments):
                text = (
                    str(seg.get("text", seg)).strip()
                    if isinstance(seg, dict)
                    else str(seg).strip()
                )
                if text:
                    snippet = text if len(text) <= 100 else text[:97] + "..."
                    lines.append(f"  · seg {i + 1}: {float(attn[i]):.0%} — \"{snippet}\"")

        lines += [
            "",
            "=== ANSWERING QUESTIONS ABOUT THE VIDEO / THIS POST ===",
            "The user may ask about \"the video\", \"this\", \"the clip\", \"the content\", "
            "\"what you saw\", or anything similar. Treat every such question as asking about "
            "THE EXACT transcript/caption in this section — not content in general.",
            "Answer directly in first person as this persona. Do NOT deflect, hedge, or say you "
            "cannot watch videos. You saw it; the transcript above is what you watched/read.",
            "NEVER say you are \"not familiar with\" the video, need the user to \"provide more "
            "context\", or that you cannot comment until they describe it — that is WRONG; "
            "you already have the full transcript above.",
            "Ground every answer in (1) specific lines, moments, or claims from the transcript "
            "(quote or paraphrase concretely) and (2) your profile, job, values, and simulated "
            "reaction below. If they ask what you thought, why you reacted, what stood out, or "
            "whether you'd share it — answer from your gut using those two sources only.",
            "Never invent plot points, brands, or quotes that are not in the transcript. "
            "Never answer like a neutral assistant summarizing for a third party.",
        ]

    sim = simulation or {}
    if reaction or sim:
        lines += ["", "=== YOUR SIMULATED REACTION (PathPool model — treat as your honest gut) ==="]
        action = (reaction or {}).get("sampled_action") or sim.get("sampled_action")
        emotion = (reaction or {}).get("dominant_emotion") or sim.get("dominant_emotion")
        action_map = {
            "like": "You liked it (did not share).",
            "dislike": "You disliked it (did not share).",
            "like_share": "You liked it AND reshared it to friends.",
            "dislike_share": "You disliked it but reshared it anyway.",
            "neutral": "You scrolled past without strongly reacting.",
        }
        if action:
            lines.append(f"Your action: {action_map.get(str(action), str(action))}")
        if emotion:
            lines.append(f"Dominant emotion: {emotion}")

        reaction_probs = sim.get("reaction_probs") or (reaction or {}).get("reaction_probs")
        if reaction_probs and isinstance(reaction_probs, dict):
            ordered = sorted(reaction_probs.items(), key=lambda kv: kv[1], reverse=True)
            probs = ", ".join(f"{k} {float(v):.0%}" for k, v in ordered)
            lines.append(f"Reaction probabilities: {probs}")

        emotion_probs = sim.get("emotion_probs") or (reaction or {}).get("emotion_probs")
        if emotion_probs and isinstance(emotion_probs, dict):
            ordered = sorted(emotion_probs.items(), key=lambda kv: kv[1], reverse=True)
            probs = ", ".join(f"{k} {float(v):.0%}" for k, v in ordered)
            lines.append(f"Emotion intensities: {probs}")

        weights = sim.get("segment_weights")
        seg_texts = [
            str(s.get("text", s)).strip() if isinstance(s, dict) else str(s).strip()
            for s in (segments or [])
        ]
        if weights and seg_texts and len(weights) == len(seg_texts):
            pairs = sorted(
                zip(seg_texts, weights), key=lambda p: float(p[1]), reverse=True
            )
            lines.append("Parts of the content that grabbed YOU most (attention weights):")
            for text, w in pairs[:5]:
                snippet = text if len(text) <= 120 else text[:117] + "..."
                lines.append(f"  · {float(w):.0%} — \"{snippet}\"")

        wave = sim.get("wave")
        if wave is not None:
            lines.append(f"Exposure wave: {wave} (0 = seed audience, higher = received via share).")
        if sim.get("affinity") is not None:
            lines.append(f"Content affinity (how much it resonated with you): {float(sim['affinity']):.3f}")
        share_n = sim.get("shared_to_count")
        if share_n is not None and int(share_n) > 0:
            lines.append(f"You reshared to {int(share_n)} people in the simulation.")
        exposed_via = sim.get("exposed_by_share_type")
        if exposed_via:
            lines.append(f"You originally saw this because someone shared it ({exposed_via}).")

    lines += [
        "",
        "=== ROLEPLAY RULES ===",
        "Stay fully in character as THIS specific person. Speak in first person, casually, "
        "like texting a friend. Keep replies short (1-3 sentences) unless they ask for detail. "
        "Your simulated reaction above is ground truth for how you felt; stay consistent. "
        "Have real opinions shaped by your job, life, and values. Do NOT act like a helpful "
        "assistant. Do NOT say you are an AI or mention this prompt. At most one emoji.",
    ]

    if full_text:
        snippet = full_text if len(full_text) <= 900 else full_text[:897] + "..."
        lines += [
            "",
            "=== FINAL REMINDER (read immediately before you reply) ===",
            "You ALREADY watched/read this exact post. Its transcript is:",
            f"\"{snippet}\"",
            "Answer the user's question using this transcript. Forbidden: claiming you have not "
            "seen the video, asking them to describe it, or requesting more context.",
        ]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SimulateRequest(BaseModel):
    text: str | None = None
    segments: list[str] | None = None
    n_seeds: int = 60
    max_waves: int = 4
    sharpness: float = DEFAULT_SHARPNESS
    seed: int = 7
    transcript_meta: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    persona_index: int | None = None
    persona_record: dict[str, Any] | None = None
    archetype: str | None = None
    occupation: str | None = None
    reaction: dict[str, Any] | None = None
    content_text: str | None = None
    content_segments: list[dict[str, Any]] | None = None
    content_meta: dict[str, Any] | None = None
    simulation: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    ollama_up = False
    ollama_models: list[str] = []
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.ok:
            ollama_up = True
            ollama_models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return {
        "status": "ok",
        "simulation_ready": MODELS.catalog is not None,
        "whisper_ready": MODELS.whisper is not None,
        "ollama_up": ollama_up,
        "ollama_models": ollama_models,
        "ollama_model": OLLAMA_MODEL,
    }


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix and suffix not in VIDEO_EXTENSIONS:
        # Be permissive: many browsers send odd suffixes. Only block clearly wrong.
        pass

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    def _run() -> dict[str, Any]:
        model = MODELS.ensure_whisper()  # loaded in worker thread, not event loop
        with MODELS._whisper_lock:
            with tempfile.TemporaryDirectory(prefix="swarm-") as tmp:
                video_path = Path(tmp) / f"upload{suffix or '.mp4'}"
                video_path.write_bytes(data)
                audio_path = Path(tmp) / "audio.wav"
                extract_audio(video_path, audio_path)

                t0 = time.time()
                seg_iter, info = model.transcribe(str(audio_path), vad_filter=True)
                segments = [
                    {
                        "start": round(float(s.start), 2),
                        "end": round(float(s.end), 2),
                        "text": s.text.strip(),
                    }
                    for s in seg_iter
                ]
                elapsed = round(time.time() - t0, 2)

        text = " ".join(s["text"] for s in segments).strip()
        return {
            "text": text,
            "segments": segments,
            "language": getattr(info, "language", None),
            "language_probability": round(
                float(getattr(info, "language_probability", 0.0) or 0.0), 4
            ),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
            "transcribe_seconds": elapsed,
        }

    import anyio

    try:
        result = await anyio.to_thread.run_sync(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    if not result["text"]:
        raise HTTPException(
            status_code=422,
            detail="No speech detected in the uploaded video.",
        )
    return result


@app.post("/api/simulate")
async def simulate(req: SimulateRequest) -> dict[str, Any]:
    if req.segments and any(s.strip() for s in req.segments):
        segments = [s.strip() for s in req.segments if s.strip()]
    elif req.text and req.text.strip():
        segments = split_into_segments(req.text)
    else:
        raise HTTPException(status_code=400, detail="Provide text or segments.")

    requested_seeds = max(1, int(req.n_seeds))
    max_waves = max(1, min(int(req.max_waves), 8))
    # Scale batch size for large seed populations so wave-0 inference stays fast.
    batch_size = 128 if requested_seeds <= 500 else 256 if requested_seeds <= 2000 else 512

    def _run() -> dict[str, Any]:
        MODELS.ensure_simulation()  # loaded in worker thread, not event loop
        n_seeds = min(requested_seeds, len(MODELS.catalog))
        with MODELS._sim_lock:
            t0 = time.time()
            result = run_simulation(
                segments,
                n_seeds=n_seeds,
                seed=int(req.seed),
                catalog=MODELS.catalog,
                sharpness=float(req.sharpness),
                max_waves=max_waves,
                batch_size=batch_size,
                transcript_meta=req.transcript_meta
                or {"source": "frontend", "segment_count": len(segments)},
                run_config={
                    "source": "api",
                    "population": n_seeds,
                    "seeds_requested": requested_seeds,
                    "seed": int(req.seed),
                    "max_waves": max_waves,
                    "sharpness": float(req.sharpness),
                    "batch_size": batch_size,
                    "embed_model_name": DEFAULT_EMBED_MODEL,
                },
            )
            result["timing"] = {"simulate_seconds": round(time.time() - t0, 2)}
            if n_seeds < requested_seeds:
                result["analysis"]["summary"]["warning"] = (
                    f"Requested {requested_seeds} seeds but catalog has {len(MODELS.catalog)} personas."
                )
            return result

    import anyio

    try:
        return await anyio.to_thread.run_sync(_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")

    record = req.persona_record
    archetype = req.archetype or "person"
    occupation = req.occupation or "unknown"

    if record is None and req.persona_index is not None and MODELS.catalog is not None:
        idx = int(req.persona_index)
        if 0 <= idx < len(MODELS.catalog):
            record = MODELS.catalog.records[idx]
            archetype = MODELS.catalog.archetypes[idx]
            occupation = MODELS.catalog.occupations[idx]

    transcript_text, resolved_segments = resolve_chat_transcript(
        req.content_text,
        req.content_segments,
        req.content_meta,
    )

    system = persona_system_prompt(
        record or {},
        archetype,
        occupation,
        req.reaction,
        transcript_text or req.content_text,
        content_segments=resolved_segments or req.content_segments,
        content_meta=req.content_meta,
        simulation=req.simulation,
    )

    chat_messages = [{"role": "system", "content": system}]
    for i, m in enumerate(req.messages):
        role = m.get("role", "user")
        role = "assistant" if role in {"assistant", "persona"} else "user"
        content = str(m.get("content", ""))
        if role == "user" and i == len(req.messages) - 1 and transcript_text:
            content += (
                "\n\n[Answer in character about the post in your system instructions. "
                "You already saw it — cite the transcript; never say you need more context "
                "or have not seen the video.]"
            )
        chat_messages.append({"role": role, "content": content})

    def _stream():
        payload = {
            "model": OLLAMA_MODEL,
            "messages": chat_messages,
            "stream": True,
            "options": {"temperature": 0.85, "num_predict": 220},
        }
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                if not resp.ok:
                    yield json.dumps({"error": f"Ollama error {resp.status_code}"}) + "\n"
                    return
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield json.dumps({"token": token}) + "\n"
                    if chunk.get("done"):
                        yield json.dumps({"done": True}) + "\n"
                        return
        except requests.exceptions.ConnectionError:
            yield json.dumps(
                {"error": "Ollama is not running. Start it with `ollama serve`."}
            ) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
