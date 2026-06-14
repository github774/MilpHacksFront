#!/usr/bin/env python3
"""Simulate content spread through a persona embedding network.

Seed a random population of n personas, predict reactions in batch, sample actions,
and propagate shares to 1-3 cosine-similar personas per share event. Personas are
resampled with replacement when the catalog is exhausted so exposure can grow past
catalog size.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from common import EMOTION_KEYS, REACTION_KEYS, load_personas
from paths import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_SHARPNESS,
    DEFAULT_SIM_OUTPUT,
    PERSONA_CSV,
    PERSONA_EMBEDDINGS_CSV,
    PERSONA_WEIGHTS,
    TRAINING_JSONL,
    rel,
)
from pipeline import embed_transcript_segments, load_pathpool, predict_batch

SHARE_ACTIONS = {"like_share", "dislike_share"}
LIKE_ACTIONS = {"like", "like_share"}
DISLIKE_ACTIONS = {"dislike", "dislike_share"}
EXPORT_FORMAT = "milphacks.network_simulation"
EXPORT_SCHEMA_VERSION = "1.2"
SIMULATION_RULES = {
    "seed_sampling": "Uniform random persona indices; with replacement when n_seeds exceeds catalog size.",
    "share_recipients_per_event": "1-3 inclusive, uniform random.",
    "share_target_selection": (
        "Up to k random indices from the 64 most cosine-similar personas "
        "(excluding source); falls back to uniform random with replacement when exhausted."
    ),
    "re_exposure": "The same persona index may appear in multiple exposure events.",
    "wave_limit": "Propagation stops after max_waves completed or when no pending shares remain.",
    "action_sampling": "One action sampled from sharpened reaction probabilities per exposure.",
}


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): _json_value(v) for k, v in row.items()}


@dataclass
class PersonaCatalog:
    """Precomputed persona embeddings + metadata for simulation."""

    embeddings: np.ndarray
    uuids: list[str]
    archetypes: list[str]
    occupations: list[str]
    records: list[dict[str, Any]]
    _similarity: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def load(
        cls,
        persona_csv: str | Path,
        embedding_csv: str | Path = PERSONA_EMBEDDINGS_CSV,
    ) -> PersonaCatalog:
        df = pd.read_csv(persona_csv)
        embeddings = load_personas(str(embedding_csv))
        if len(df) != len(embeddings):
            raise ValueError(
                f"persona rows ({len(df)}) != embedding rows ({len(embeddings)})"
            )
        embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9
        )
        uuids = df["uuid"].astype(str).tolist()
        archetypes = [_archetype_label(row) for _, row in df.iterrows()]
        occupations = [
            str(row.get("occupation", "unknown"))
            if pd.notna(row.get("occupation"))
            else "unknown"
            for _, row in df.iterrows()
        ]
        records = [_row_to_dict(row) for _, row in df.iterrows()]
        return cls(
            embeddings=embeddings.astype(np.float32),
            uuids=uuids,
            archetypes=archetypes,
            occupations=occupations,
            records=records,
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self), size=n, replace=n > len(self))

    def random_indices(self, k: int, rng: np.random.Generator) -> list[int]:
        return [int(i) for i in rng.choice(len(self), size=k, replace=True)]

    def similar_indices(
        self,
        source_idx: int,
        k: int,
        excluded: set[int],
        rng: np.random.Generator,
        *,
        pool: int = 64,
    ) -> list[int]:
        """Pick up to k random indices from the most similar personas."""
        if self._similarity is None:
            self._similarity = self.embeddings @ self.embeddings.T

        sims = self._similarity[source_idx].copy()
        for idx in excluded:
            sims[idx] = -np.inf
        sims[source_idx] = -np.inf

        top = np.argpartition(sims, -pool)[-pool:]
        top = top[np.argsort(sims[top])[::-1]]
        candidates = [int(i) for i in top if sims[i] > -np.inf]
        if not candidates:
            return self.random_indices(k, rng)
        pick = min(k, len(candidates))
        chosen = rng.choice(candidates, size=pick, replace=False)
        return [int(i) for i in chosen]


def _archetype_label(row: pd.Series) -> str:
    occ = row.get("occupation")
    age = row.get("age")
    sex = row.get("sex")
    if pd.notna(occ) and str(occ).strip():
        parts = [str(occ).replace("_", " ")]
        if pd.notna(age):
            parts.append(f"age {int(age)}")
        if pd.notna(sex):
            parts.append(str(sex).lower())
        return ", ".join(parts)
    persona = row.get("persona") or row.get("professional_persona") or ""
    if pd.notna(persona) and str(persona).strip():
        text = str(persona).strip()
        return text[:80] + ("..." if len(text) > 80 else "")
    return str(row.get("uuid", "unknown"))


def _is_reportable_archetype(archetype: str) -> bool:
    """Skip children / non-workers when ranking archetypes."""
    return not archetype.lower().startswith("not in workforce")


def _top_archetypes(counter: Counter, n: int = 3) -> list[dict[str, Any]]:
    filtered = Counter(
        {k: v for k, v in counter.items() if _is_reportable_archetype(k)}
    )
    return [{"archetype": k, "count": v} for k, v in filtered.most_common(n)]


def _top_counter(counter: Counter, n: int = 5, *, key: str = "label") -> list[dict[str, Any]]:
    return [{key: k, "count": v} for k, v in counter.most_common(n)]


def _round_prob_dict(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in probs.items()}


def _mean_prob_dicts(
    agents: list[dict[str, Any]],
    prob_key: str,
    keys: list[str],
) -> dict[str, float]:
    if not agents:
        return {k: 0.0 for k in keys}
    arr = np.array(
        [[a[prob_key][k] for k in keys] for a in agents],
        dtype=np.float64,
    )
    return _round_prob_dict({k: float(arr[:, i].mean()) for i, k in enumerate(keys)})


def _dominant_key(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def _compute_advanced_analysis(
    agents: list[dict[str, Any]],
    share_edges: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate emotion, virality, and share-pattern metrics from simulation output."""
    seeds_exposed = summary["seeds_exposed"]
    total_exposed = summary["total_exposed"]
    share_events = summary["share_events"]
    share_recipients = summary["share_recipients"]
    share_recipients_simulated = summary["share_recipients_simulated"]
    viral_exposed = summary["viral_exposed"]

    sharers = [a for a in agents if a["sampled_action"] in SHARE_ACTIONS]
    non_sharers = [a for a in agents if a["sampled_action"] not in SHARE_ACTIONS]
    viral_agents = [a for a in agents if a["wave"] > 0]
    seed_agents = [a for a in agents if a["wave"] == 0]

    mean_emotions = _mean_prob_dicts(agents, "emotion_probs", EMOTION_KEYS)
    mean_reactions = _mean_prob_dicts(agents, "reaction_probs", REACTION_KEYS)

    wave_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for agent in agents:
        wave_groups[agent["wave"]].append(agent)
    emotions_by_wave = {
        str(wave): _mean_prob_dicts(group, "emotion_probs", EMOTION_KEYS)
        for wave, group in sorted(wave_groups.items())
    }

    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent in agents:
        action_groups[agent["sampled_action"]].append(agent)
    emotions_by_action = {
        action: _mean_prob_dicts(group, "emotion_probs", EMOTION_KEYS)
        for action, group in sorted(action_groups.items())
    }

    share_type_counts = Counter(e["share_type"] for e in share_edges)
    shares_by_wave = Counter(e["wave"] for e in share_edges)
    exposures_by_wave = Counter(a["wave"] for a in agents)
    recipients_per_share = Counter(
        len(a["shared_to_indices"]) for a in sharers if a["shared_to_indices"]
    )
    inbound_share_types = Counter(
        a["exposed_by_share_type"]
        for a in viral_agents
        if a.get("exposed_by_share_type")
    )
    sharing_archetypes = Counter(e["from_archetype"] for e in share_edges)
    recipient_archetypes = Counter(e["to_archetype"] for e in share_edges)
    occupation_flow = Counter(
        (e["from_occupation"], e["to_occupation"]) for e in share_edges
    )

    segment_count = max((len(a.get("segment_weights") or []) for a in agents), default=0)
    mean_segment_weights_all: list[float] = []
    mean_segment_weights_sharers: list[float] = []
    if segment_count:
        all_weights = np.array(
            [a.get("segment_weights") or [0.0] * segment_count for a in agents],
            dtype=np.float64,
        )
        mean_segment_weights_all = [
            round(float(v), 4) for v in all_weights.mean(axis=0).tolist()
        ]
        if sharers:
            sharer_weights = np.array(
                [
                    a.get("segment_weights") or [0.0] * segment_count
                    for a in sharers
                ],
                dtype=np.float64,
            )
            mean_segment_weights_sharers = [
                round(float(v), 4) for v in sharer_weights.mean(axis=0).tolist()
            ]

    def _mean_metric(group: list[dict[str, Any]], key: str) -> float | None:
        if not group:
            return None
        return round(float(np.mean([a[key] for a in group])), 4)

    predicted_share_prob = round(
        float(
            np.mean(
                [
                    a["reaction_probs"]["like_share"]
                    + a["reaction_probs"]["dislike_share"]
                    for a in agents
                ]
            )
        ),
        4,
    )
    observed_share_rate = round(share_events / total_exposed, 4) if total_exposed else 0.0
    amplification = round(total_exposed / seeds_exposed, 4) if seeds_exposed else 0.0
    viral_reach_ratio = (
        round(viral_exposed / total_exposed, 4) if total_exposed else 0.0
    )
    effective_branching = (
        round(share_recipients_simulated / share_events, 4) if share_events else 0.0
    )
    secondary_share_rate = (
        round(
            sum(1 for a in sharers if a["wave"] > 0) / len(viral_agents),
            4,
        )
        if viral_agents
        else 0.0
    )
    like_share_ratio = (
        round(
            summary["like_shares"]
            / (summary["like_shares"] + summary["dislike_shares"]),
            4,
        )
        if summary["like_shares"] + summary["dislike_shares"]
        else None
    )
    max_share_wave = max((e["wave"] for e in share_edges), default=None)
    max_exposure_wave = max((a["wave"] for a in agents), default=0)

    sharer_emotions = _mean_prob_dicts(sharers, "emotion_probs", EMOTION_KEYS)
    non_sharer_emotions = _mean_prob_dicts(non_sharers, "emotion_probs", EMOTION_KEYS)
    emotion_share_lift = {
        k: round(sharer_emotions[k] - non_sharer_emotions[k], 4)
        for k in EMOTION_KEYS
    }

    like_agents = [a for a in agents if a["sampled_action"] in LIKE_ACTIONS]
    dislike_agents = [a for a in agents if a["sampled_action"] in DISLIKE_ACTIONS]
    like_emotions = _mean_prob_dicts(like_agents, "emotion_probs", EMOTION_KEYS)
    dislike_emotions = _mean_prob_dicts(dislike_agents, "emotion_probs", EMOTION_KEYS)
    emotion_polarization = {
        k: round(like_emotions[k] - dislike_emotions[k], 4) for k in EMOTION_KEYS
    }

    share_rate_by_wave = {
        str(wave): round(
            sum(1 for a in group if a["sampled_action"] in SHARE_ACTIONS) / len(group),
            4,
        )
        for wave, group in sorted(wave_groups.items())
        if group
    }

    engaged = summary["likes"] + summary["dislikes"] + summary["like_shares"] + summary["dislike_shares"]
    polarization_index = (
        round((summary["dislikes"] + summary["dislike_shares"] - summary["likes"] - summary["like_shares"]) / engaged, 4)
        if engaged
        else 0.0
    )

    exposure_by_index = {a["persona_index"]: a for a in agents}
    chain_depth = 0
    for edge in share_edges:
        depth = 1
        cursor = edge["from_index"]
        seen: set[int] = {edge["to_index"], cursor}
        while True:
            parent = exposure_by_index.get(cursor)
            if parent is None or parent.get("exposed_by_index") is None:
                break
            cursor = parent["exposed_by_index"]
            if cursor in seen:
                break
            seen.add(cursor)
            depth += 1
        chain_depth = max(chain_depth, depth)

    return {
        "derived_from": ["raw.agents", "raw.share_edges"],
        "emotions": {
            "mean_probs": mean_emotions,
            "dominant_emotion": _dominant_key(mean_emotions),
            "mean_probs_by_wave": emotions_by_wave,
            "mean_probs_by_action": emotions_by_action,
            "mean_probs_sharers": _mean_prob_dicts(sharers, "emotion_probs", EMOTION_KEYS),
            "mean_probs_non_sharers": _mean_prob_dicts(
                non_sharers, "emotion_probs", EMOTION_KEYS
            ),
            "mean_probs_seeds": _mean_prob_dicts(seed_agents, "emotion_probs", EMOTION_KEYS),
            "mean_probs_viral": _mean_prob_dicts(viral_agents, "emotion_probs", EMOTION_KEYS),
            "top_emotions_among_sharers": _top_counter(
                Counter(_dominant_key(a["emotion_probs"]) for a in sharers),
                n=5,
                key="emotion",
            ),
        },
        "reactions": {
            "mean_probs": mean_reactions,
            "dominant_reaction": _dominant_key(mean_reactions),
            "predicted_share_probability": predicted_share_prob,
            "observed_share_rate": observed_share_rate,
            "share_rate_delta": round(observed_share_rate - predicted_share_prob, 4),
        },
        "virality": {
            "amplification_factor": amplification,
            "viral_reach_ratio": viral_reach_ratio,
            "share_conversion_rate": observed_share_rate,
            "effective_branching_factor": effective_branching,
            "secondary_share_rate": secondary_share_rate,
            "like_share_ratio": like_share_ratio,
            "max_share_wave": max_share_wave,
            "max_exposure_wave": max_exposure_wave,
            "cascade_depth": max_exposure_wave,
            "share_recipients_per_seed": (
                round(share_recipients / seeds_exposed, 4) if seeds_exposed else 0.0
            ),
            "simulated_recipients_per_seed": (
                round(share_recipients_simulated / seeds_exposed, 4)
                if seeds_exposed
                else 0.0
            ),
        },
        "share_patterns": {
            "share_type_counts": dict(share_type_counts),
            "shares_by_wave": {str(k): v for k, v in sorted(shares_by_wave.items())},
            "exposures_by_wave": {
                str(k): v for k, v in sorted(exposures_by_wave.items())
            },
            "recipients_per_share": {str(k): v for k, v in sorted(recipients_per_share.items())},
            "inbound_share_types": dict(inbound_share_types),
            "top_sharing_archetypes": _top_archetypes(sharing_archetypes, n=5),
            "top_recipient_archetypes": _top_archetypes(recipient_archetypes, n=5),
            "top_occupation_flows": [
                {
                    "from_occupation": pair[0],
                    "to_occupation": pair[1],
                    "count": count,
                }
                for pair, count in occupation_flow.most_common(10)
            ],
            "unique_sharers": len({e["from_index"] for e in share_edges}),
            "unique_recipients": len({e["to_index"] for e in share_edges}),
            "reshare_events": sum(1 for a in sharers if a["wave"] > 0),
        },
        "affinity": {
            "mean_all": _mean_metric(agents, "affinity"),
            "mean_sharers": _mean_metric(sharers, "affinity"),
            "mean_non_sharers": _mean_metric(non_sharers, "affinity"),
            "mean_viral": _mean_metric(viral_agents, "affinity"),
            "mean_affinity_s_all": _mean_metric(agents, "affinity_s"),
            "mean_affinity_s_sharers": _mean_metric(sharers, "affinity_s"),
        },
        "segment_attention": {
            "segment_count": segment_count,
            "mean_weights_all": mean_segment_weights_all,
            "mean_weights_sharers": mean_segment_weights_sharers,
        },
        "insights": {
            "emotion_share_lift": emotion_share_lift,
            "top_share_lift_emotions": [
                {"emotion": k, "lift": v}
                for k, v in sorted(
                    emotion_share_lift.items(), key=lambda kv: kv[1], reverse=True
                )[:3]
            ],
            "emotion_polarization_like_minus_dislike": emotion_polarization,
            "polarization_index": polarization_index,
            "share_rate_by_wave": share_rate_by_wave,
            "longest_share_chain_depth": chain_depth,
            "virality_score": round(
                amplification * observed_share_rate * (effective_branching or 0.0),
                4,
            ),
        },
    }


def _sample_action(probs: dict[str, float], rng: np.random.Generator) -> str:
    keys = REACTION_KEYS
    p = np.array([probs[k] for k in keys], dtype=np.float64)
    p = p / p.sum()
    return keys[int(rng.choice(len(keys), p=p))]


def _load_transcript_segments(
    *,
    segment_texts: list[str] | None,
    transcript_jsonl: str | None,
    record_index: int | None,
    record_id: str | None,
) -> tuple[list[str], dict[str, Any]]:
    if segment_texts:
        segments = [s.strip() for s in segment_texts if s.strip()]
        meta = {
            "source": "inline_segments",
            "segment_count": len(segments),
            "segments": [
                {"index": i, "text": text}
                for i, text in enumerate(segments)
            ],
        }
        return segments, meta

    if not transcript_jsonl:
        raise ValueError("provide segment_texts or --transcript-jsonl")

    with open(transcript_jsonl, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    if record_id:
        record = next(r for r in records if r.get("id") == record_id)
    elif record_index is not None:
        record = records[record_index]
    else:
        record = records[0]

    segments = [s["text"].strip() for s in record["segments"]]
    meta = {
        "source": transcript_jsonl,
        "record_index": record_index,
        "record_id": record_id,
        "id": record.get("id"),
        "topic": record.get("topic"),
        "tone": record.get("tone"),
        "language": record.get("language"),
        "language_probability": record.get("language_probability"),
        "duration": record.get("duration"),
        "segment_count": len(segments),
        "text": record.get("text"),
        "segments": [
            {
                "index": i,
                "text": seg["text"].strip(),
                "start": seg.get("start"),
                "end": seg.get("end"),
            }
            for i, seg in enumerate(record["segments"])
        ],
        "record": record,
    }
    return segments, meta


def _predict_in_chunks(
    seg_emb: np.ndarray,
    persona_indices: np.ndarray,
    catalog: PersonaCatalog,
    model,
    device,
    *,
    batch_size: int,
    sharpness: float,
) -> list[dict]:
    results: list[dict] = []
    for start in range(0, len(persona_indices), batch_size):
        chunk = persona_indices[start : start + batch_size]
        embs = catalog.embeddings[chunk]
        results.extend(
            predict_batch(
                seg_emb,
                embs,
                model,
                device,
                sharpness=sharpness,
            )
        )
    return results


def _referenced_persona_indices(result: dict[str, Any]) -> set[int]:
    indices: set[int] = set(result.get("seed_indices", []))
    for agent in result["agents"]:
        indices.add(agent["persona_index"])
        if agent.get("exposed_by_index") is not None:
            indices.add(agent["exposed_by_index"])
        indices.update(agent.get("shared_to_indices", []))
    for edge in result["share_edges"]:
        indices.add(edge["from_index"])
        indices.add(edge["to_index"])
    return indices


def _export_personas(catalog: PersonaCatalog, indices: set[int]) -> dict[str, dict[str, Any]]:
    personas: dict[str, dict[str, Any]] = {}
    for idx in sorted(indices):
        personas[str(idx)] = {
            "persona_index": idx,
            "uuid": catalog.uuids[idx],
            "archetype": catalog.archetypes[idx],
            "occupation": catalog.occupations[idx],
            "record": catalog.records[idx],
        }
    return personas


def _export_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop inline persona blobs; full records live in raw.personas."""
    return [{k: v for k, v in agent.items() if k != "persona"} for agent in agents]


def _portable_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        return str(value)
    try:
        return rel(path)
    except ValueError:
        return str(path)


def _build_run_config(run_config: dict[str, Any] | None, **defaults: Any) -> dict[str, Any]:
    export_config = dict(run_config or {})
    for key, value in defaults.items():
        if key not in export_config or export_config[key] is None:
            export_config[key] = value
    for key, value in list(export_config.items()):
        if isinstance(value, (str, Path)):
            export_config[key] = _portable_path(value)
    return export_config


def _finalize_export(
    result: dict[str, Any],
    *,
    catalog: PersonaCatalog,
    segment_texts: list[str],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    referenced = _referenced_persona_indices(result)
    transcript = dict(result.get("transcript") or {})
    transcript.setdefault("segment_count", len(segment_texts))
    transcript.setdefault(
        "segments",
        [{"index": i, "text": text} for i, text in enumerate(segment_texts)],
    )

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": EXPORT_FORMAT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": {
            "reaction_keys": REACTION_KEYS,
            "emotion_keys": EMOTION_KEYS,
            "share_actions": sorted(SHARE_ACTIONS),
            "like_actions": sorted(LIKE_ACTIONS),
            "dislike_actions": sorted(DISLIKE_ACTIONS),
        },
        "simulation_rules": SIMULATION_RULES,
        "run_config": run_config,
        "catalog": {
            "size": len(catalog),
            "persona_csv": _portable_path(run_config.get("persona_csv")),
            "embedding_csv": _portable_path(run_config.get("embedding_csv")),
            "referenced_persona_count": len(referenced),
        },
        "config": result["config"],
        "raw": {
            "transcript": transcript,
            "personas": _export_personas(catalog, referenced),
            "seed_indices": result["seed_indices"],
            "seed_indices_requested": result["seed_indices_requested"],
            "agents": _export_agents(result["agents"]),
            "share_edges": result["share_edges"],
        },
        "analysis": {
            "summary": result["summary"],
            "top_liked_archetypes": result["top_liked_archetypes"],
            "top_disliked_archetypes": result["top_disliked_archetypes"],
            "advanced": _compute_advanced_analysis(
                result["agents"],
                result["share_edges"],
                result["summary"],
            ),
        },
    }


def run_simulation(
    segment_texts: list[str],
    *,
    n_seeds: int = 50,
    seed: int = 7,
    catalog: PersonaCatalog | None = None,
    persona_csv: str | Path = PERSONA_CSV,
    embedding_csv: str | Path = PERSONA_EMBEDDINGS_CSV,
    weights_path: str | Path = PERSONA_WEIGHTS,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    device: str | None = None,
    sharpness: float = DEFAULT_SHARPNESS,
    batch_size: int = 128,
    max_waves: int = 5,
    transcript_meta: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    catalog = catalog or PersonaCatalog.load(persona_csv, embedding_csv)

    embed_model = SentenceTransformer(embed_model_name, device=device)
    pathpool, torch_device = load_pathpool(weights_path, device=device)
    seg_emb = embed_transcript_segments(segment_texts, model=embed_model)

    seed_indices = catalog.sample_indices(n_seeds, rng)
    seeds_requested = int(n_seeds)

    exposed: set[int] = set()  # unique personas seen (for stats only)
    pending_waves: list[list[tuple[int, int | None, str | None]]] = [
        [(int(i), None, None) for i in seed_indices]
    ]

    agents: list[dict[str, Any]] = []
    share_edges: list[dict[str, Any]] = []
    action_counts: Counter = Counter()
    waves_completed = 0
    share_recipients_dropped = 0
    exposure_id = 0

    while pending_waves and waves_completed <= max_waves:
        wave = pending_waves.pop(0)
        new_entries = list(wave)
        if not new_entries:
            continue

        indices = np.array([e[0] for e in new_entries], dtype=np.int64)
        wave_num = waves_completed

        preds = _predict_in_chunks(
            seg_emb,
            indices,
            catalog,
            pathpool,
            torch_device,
            batch_size=batch_size,
            sharpness=sharpness,
        )

        next_wave: list[tuple[int, int | None, str | None]] = []

        for (persona_idx, exposed_by, share_type), pred in zip(new_entries, preds):
            exposed.add(persona_idx)
            action = _sample_action(pred["reactions"], rng)
            action_counts[action] += 1

            shared_to: list[int] = []
            if action in SHARE_ACTIONS:
                n_share = int(rng.integers(1, 4))  # 1-3 inclusive
                shared_to = catalog.similar_indices(
                    persona_idx, n_share, exposed, rng
                )
                for target in shared_to:
                    share_edges.append(
                        {
                            "from_exposure_id": exposure_id,
                            "from_index": persona_idx,
                            "from_uuid": catalog.uuids[persona_idx],
                            "from_archetype": catalog.archetypes[persona_idx],
                            "from_occupation": catalog.occupations[persona_idx],
                            "to_index": target,
                            "to_uuid": catalog.uuids[target],
                            "to_archetype": catalog.archetypes[target],
                            "to_occupation": catalog.occupations[target],
                            "share_type": action,
                            "wave": wave_num,
                        }
                    )
                    next_wave.append((target, persona_idx, action))

            agents.append(
                {
                    "exposure_id": exposure_id,
                    "persona_index": persona_idx,
                    "uuid": catalog.uuids[persona_idx],
                    "archetype": catalog.archetypes[persona_idx],
                    "occupation": catalog.occupations[persona_idx],
                    "persona": catalog.records[persona_idx],
                    "wave": wave_num,
                    "exposed_by_index": exposed_by,
                    "exposed_by_uuid": (
                        catalog.uuids[exposed_by] if exposed_by is not None else None
                    ),
                    "exposed_by_share_type": share_type,
                    "reaction_probs": pred["reactions"],
                    "emotion_probs": pred["emotions"],
                    "segment_weights": pred.get("segment_weights", []),
                    "affinity": pred["affinity"],
                    "affinity_s": pred["affinity_s"],
                    "sampled_action": action,
                    "shared_to_indices": shared_to,
                    "shared_to_uuids": [catalog.uuids[i] for i in shared_to],
                }
            )
            exposure_id += 1

        if next_wave:
            pending_waves.append(next_wave)
        waves_completed += 1

    queued = [idx for wave in pending_waves for idx, _, _ in wave]
    share_recipients_dropped += len(queued)

    seeds_exposed = sum(1 for a in agents if a["wave"] == 0)
    viral_exposed = len(agents) - seeds_exposed
    share_events = sum(action_counts[a] for a in SHARE_ACTIONS)
    share_recipients = len(share_edges)
    agent_indices = {a["persona_index"] for a in agents}
    simulated_recipients = sum(
        1 for e in share_edges if e["to_index"] in agent_indices
    )

    likes = action_counts["like"]
    dislikes = action_counts["dislike"]
    like_shares = action_counts["like_share"]
    dislike_shares = action_counts["dislike_share"]
    neutral = action_counts["neutral"]

    liked_archetypes = Counter(
        a["archetype"] for a in agents if a["sampled_action"] in LIKE_ACTIONS
    )
    disliked_archetypes = Counter(
        a["archetype"] for a in agents if a["sampled_action"] in DISLIKE_ACTIONS
    )

    summary = {
        "total_exposed": len(agents),
        "unique_personas_exposed": len(exposed),
        "seeds_requested": seeds_requested,
        "seeds_exposed": seeds_exposed,
        "viral_exposed": viral_exposed,
        "waves_completed": waves_completed,
        "likes": likes,
        "dislikes": dislikes,
        "like_shares": like_shares,
        "dislike_shares": dislike_shares,
        "share_events": share_events,
        "share_recipients": share_recipients,
        "share_recipients_simulated": simulated_recipients,
        "share_recipients_dropped": share_recipients_dropped,
        "neutral": neutral,
        "action_counts": dict(action_counts),
    }
    result = {
        "summary": summary,
        "top_liked_archetypes": _top_archetypes(liked_archetypes),
        "top_disliked_archetypes": _top_archetypes(disliked_archetypes),
        "transcript": transcript_meta or {"segment_count": len(segment_texts)},
        "seed_indices": seed_indices.tolist(),
        "seed_indices_requested": int(seeds_requested),
        "agents": agents,
        "share_edges": share_edges,
        "config": {
            "n_seeds": n_seeds,
            "seed": seed,
            "max_waves": max_waves,
            "batch_size": batch_size,
            "sharpness": sharpness,
            "embed_model_name": embed_model_name,
            "weights_path": str(weights_path),
            "persona_csv": str(persona_csv),
            "embedding_csv": str(embedding_csv),
            "device": device,
        },
    }
    export_config = _build_run_config(
        run_config,
        embed_model_name=embed_model_name,
        weights_path=rel(weights_path),
        persona_csv=rel(persona_csv),
        embedding_csv=rel(embedding_csv),
        device=device,
        sharpness=sharpness,
        batch_size=batch_size,
        max_waves=max_waves,
        n_seeds=n_seeds,
        seed=seed,
    )
    return _finalize_export(
        result,
        catalog=catalog,
        segment_texts=segment_texts,
        run_config=export_config,
    )


def print_summary(result: dict[str, Any]) -> None:
    analysis = result.get("analysis", result)
    raw = result.get("raw", result)
    s = analysis["summary"]
    print("=== Network simulation summary ===")
    print(f"  Total exposed:           {s['total_exposed']}")
    print(f"  Unique personas:         {s['unique_personas_exposed']}")
    print(f"  Seeds requested/exposed: {s['seeds_requested']} / {s['seeds_exposed']}")
    print(f"  Viral (from shares):     {s['viral_exposed']}")
    print(f"  Waves completed:         {s['waves_completed']}")
    print(f"  Likes:                   {s['likes']}")
    print(f"  Dislikes:                {s['dislikes']}")
    print(f"  Like-shares:             {s['like_shares']}")
    print(f"  Dislike-shares:          {s['dislike_shares']}")
    print(f"  Share events:            {s['share_events']}")
    print(f"  Share recipients:        {s['share_recipients']} "
          f"({s['share_recipients_simulated']} simulated, "
          f"{s['share_recipients_dropped']} dropped)")
    print(f"  Neutral/pass:            {s['neutral']}")
    if s.get("warning"):
        print(f"\n  WARNING: {s['warning']}")
    ac = s["action_counts"]
    assert sum(ac.values()) == s["total_exposed"], ac
    print("\nTop liked archetypes:")
    for row in analysis["top_liked_archetypes"]:
        print(f"  {row['count']:3d}  {row['archetype']}")
    print("\nTop disliked archetypes:")
    for row in analysis["top_disliked_archetypes"]:
        print(f"  {row['count']:3d}  {row['archetype']}")
    print(f"\nShare edges: {len(raw['share_edges'])}")
    advanced = analysis.get("advanced") or analysis.get("advanced_analysis")
    if advanced:
        emo = advanced["emotions"]
        vir = advanced["virality"]
        pat = advanced["share_patterns"]
        insights = advanced.get("insights", {})
        print("\n=== Advanced analysis ===")
        print(f"  Dominant emotion:        {emo['dominant_emotion']}")
        print(f"  Predicted share prob:    {advanced['reactions']['predicted_share_probability']}")
        print(f"  Observed share rate:     {advanced['reactions']['observed_share_rate']}")
        print(f"  Amplification factor:    {vir['amplification_factor']}")
        print(f"  Effective branching:     {vir['effective_branching_factor']}")
        print(f"  Cascade depth:           {vir['cascade_depth']}")
        print(f"  Unique sharers:          {pat['unique_sharers']}")
        print(f"  Reshare events:          {pat['reshare_events']}")
        if insights:
            print(f"  Virality score:          {insights.get('virality_score')}")
            print(f"  Polarization index:      {insights.get('polarization_index')}")
            print(f"  Longest share chain:     {insights.get('longest_share_chain_depth')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simulate transcript propagation through a persona network."
    )
    ap.add_argument(
        "--transcript-jsonl",
        default=str(TRAINING_JSONL),
        help="JSONL with transcript segments.",
    )
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--record-id", default=None, help="Pick transcript by id.")
    ap.add_argument(
        "-n",
        "--population",
        type=int,
        default=50,
        help="Seed population size.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--persona-csv", default=str(PERSONA_CSV))
    ap.add_argument("--embedding-csv", default=str(PERSONA_EMBEDDINGS_CSV))
    ap.add_argument("--weights", default=str(PERSONA_WEIGHTS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-waves", type=int, default=5)
    ap.add_argument("--sharpness", type=float, default=DEFAULT_SHARPNESS)
    ap.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_SIM_OUTPUT),
        help="Write full simulation JSON here.",
    )
    args = ap.parse_args()

    segments, meta = _load_transcript_segments(
        segment_texts=None,
        transcript_jsonl=args.transcript_jsonl,
        record_index=args.record_index,
        record_id=args.record_id,
    )

    result = run_simulation(
        segments,
        n_seeds=args.population,
        seed=args.seed,
        persona_csv=args.persona_csv,
        embedding_csv=args.embedding_csv,
        weights_path=args.weights,
        device=args.device,
        sharpness=args.sharpness,
        batch_size=args.batch_size,
        max_waves=args.max_waves,
        transcript_meta=meta,
        run_config={
            "transcript_jsonl": args.transcript_jsonl,
            "record_index": args.record_index,
            "record_id": args.record_id,
            "population": args.population,
            "seed": args.seed,
            "persona_csv": args.persona_csv,
            "embedding_csv": args.embedding_csv,
            "weights_path": args.weights,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_waves": args.max_waves,
            "sharpness": args.sharpness,
            "output": args.output,
            "embed_model_name": DEFAULT_EMBED_MODEL,
        },
    )

    print_summary(result)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
