#!/usr/bin/env python3
"""Generate a synthetic training dataset of short video-ad transcripts plus
audience reactions and emotion scalars.

Each record matches the transcript schema in transcripts/video.json
(language, language_probability, duration, text, segments) and adds:

  reactions: probability distribution over audience actions that sums to 1.0
    - like
    - dislike
    - like_share        (liked it AND reshared)
    - dislike_share     (disliked it AND reshared / hate-shared)
    - neutral           (watched, no action)

  emotions: scalar 0..1 intensities the content evokes
    - empathy           (felt-for / human-warmth response)
    - relation          (relatability / "this is about me")
    - inspiration
    - curiosity
    - joy

The transcript tone (positive / neutral / negative) is sampled per record and
drives both the wording AND the reaction distribution, so the labels stay
coherent with the text. The dataset deliberately spans glowing positive ads,
flat informational ones, and pushy / spammy negative ones.

Usage:
    python3 build_training_dataset.py
    python3 build_training_dataset.py --count 1000 --output transcripts/training_data.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path

from paths import TRAINING_JSONL

DEFAULT_COUNT = 1000
DEFAULT_OUTPUT = str(TRAINING_JSONL)
WORDS_PER_SECOND = 2.7  # rough conversational ad pacing
SENTENCES_PER_SEGMENT = 2


# --------------------------------------------------------------------------- #
# Topic catalog. Each topic has product-name fragments and a human "subject"
# used to personalize lines. emotional_pull biases empathy/inspiration.
# --------------------------------------------------------------------------- #
TOPICS = [
    {"name": "fitness_app", "noun": "fitness app", "subject": "your workouts",
     "audience": "runners and lifters", "emotional_pull": 0.45},
    {"name": "meal_kit", "noun": "meal kit", "subject": "weeknight dinners",
     "audience": "busy home cooks", "emotional_pull": 0.4},
    {"name": "travel_app", "noun": "travel app", "subject": "your next trip",
     "audience": "weekend travelers", "emotional_pull": 0.5},
    {"name": "finance_app", "noun": "budgeting app", "subject": "your money",
     "audience": "people paying down debt", "emotional_pull": 0.55},
    {"name": "ai_tool", "noun": "AI assistant", "subject": "your busywork",
     "audience": "developers and writers", "emotional_pull": 0.25},
    {"name": "gardening", "noun": "raised-bed kit", "subject": "your backyard garden",
     "audience": "weekend gardeners", "emotional_pull": 0.45},
    {"name": "music_learning", "noun": "guitar course", "subject": "learning an instrument",
     "audience": "aspiring musicians", "emotional_pull": 0.6},
    {"name": "language_app", "noun": "language app", "subject": "learning a new language",
     "audience": "language learners", "emotional_pull": 0.5},
    {"name": "meditation", "noun": "meditation app", "subject": "your stress",
     "audience": "anyone feeling burned out", "emotional_pull": 0.75},
    {"name": "pet_care", "noun": "fresh dog food", "subject": "your dog",
     "audience": "dog owners", "emotional_pull": 0.7},
    {"name": "skincare", "noun": "skincare serum", "subject": "your skin",
     "audience": "people chasing clear skin", "emotional_pull": 0.35},
    {"name": "outdoor_gear", "noun": "hiking pack", "subject": "your next hike",
     "audience": "hikers and campers", "emotional_pull": 0.5},
    {"name": "coffee", "noun": "coffee subscription", "subject": "your morning cup",
     "audience": "coffee lovers", "emotional_pull": 0.4},
    {"name": "online_course", "noun": "online course", "subject": "a new career skill",
     "audience": "career switchers", "emotional_pull": 0.55},
    {"name": "photography", "noun": "camera preset pack", "subject": "your photos",
     "audience": "hobby photographers", "emotional_pull": 0.45},
    {"name": "home_diy", "noun": "smart toolkit", "subject": "your home projects",
     "audience": "weekend DIYers", "emotional_pull": 0.35},
    {"name": "parenting", "noun": "kids learning app", "subject": "your kids",
     "audience": "parents of young kids", "emotional_pull": 0.7},
    {"name": "ev_car", "noun": "electric SUV", "subject": "your commute",
     "audience": "first-time EV buyers", "emotional_pull": 0.4},
    {"name": "supplements", "noun": "daily greens powder", "subject": "your energy",
     "audience": "people chasing more energy", "emotional_pull": 0.3},
    {"name": "small_biz", "noun": "invoicing tool", "subject": "your small business",
     "audience": "freelancers and founders", "emotional_pull": 0.4},
    {"name": "sleep_tech", "noun": "cooling mattress", "subject": "your sleep",
     "audience": "restless sleepers", "emotional_pull": 0.6},
    {"name": "budget_travel", "noun": "flight-deals service", "subject": "cheap flights",
     "audience": "deal hunters", "emotional_pull": 0.45},
    {"name": "streaming", "noun": "documentary streaming service", "subject": "your watchlist",
     "audience": "documentary fans", "emotional_pull": 0.5},
    {"name": "ecofriendly", "noun": "refillable cleaning kit", "subject": "your home",
     "audience": "eco-conscious shoppers", "emotional_pull": 0.55},
]

BRAND_PREFIX = ["Nova", "Bright", "Pulse", "Ever", "Lumen", "Peak", "Drift", "Aura",
                "Kindred", "Northwind", "Tidal", "Forge", "Maple", "Ridge", "Sol",
                "Verve", "Hatch", "Bloom", "Atlas", "Echo", "Glow", "Orbit", "Vale",
                "Crest", "Harbor", "Stride", "Ember", "Clover", "Beacon", "Quill"]
BRAND_SUFFIX = ["ly", "io", "ify", "well", "go", "lab", "hub", "kit", "co", "wave",
                "mind", "fuel", "nest", "path", "core", "spark", "loop", "base"]


# --------------------------------------------------------------------------- #
# Tone-tiered sentence pools. {b}=brand, {noun}, {subject}, {audience}.
# --------------------------------------------------------------------------- #
POSITIVE = {
    "hook": [
        "I almost cried the first time {b} actually worked for me.",
        "Six months ago I was ready to give up on {subject}.",
        "Nobody told me {subject} could feel this good.",
        "This is the {noun} I wish I'd found years ago.",
        "If {subject} has been wearing you down, please watch this.",
        "{b} quietly changed how I think about {subject}.",
    ],
    "body": [
        "It just gets out of your way and lets you focus.",
        "Everything is one tap, no manual, no setup headache.",
        "Real people built this for {audience}, and you can feel it.",
        "It learns what you need and meets you there.",
        "No gimmicks, no upsells, just something that works.",
        "The little details show somebody actually cared.",
        "It turned a chore I dreaded into part of my routine I love.",
    ],
    "proof": [
        "Thousands of {audience} already swear by {b}.",
        "The reviews aren't hype, they're people whose lives got easier.",
        "My friends kept asking what changed, and the answer was {b}.",
        "I've recommended it to everyone I care about.",
    ],
    "cta": [
        "Try {b} today and see for yourself.",
        "Give {b} one week, that's all it takes.",
        "Start free with {b}, no card needed.",
        "Your future self will thank you for trying {b}.",
    ],
}

NEUTRAL = {
    "hook": [
        "{b} is a {noun} for {audience}.",
        "Here's how {b} handles {subject}.",
        "Let's walk through what {b} does.",
        "{b} aims to simplify {subject}.",
        "A quick look at {b} and how it works.",
    ],
    "body": [
        "It covers the basics most {audience} look for.",
        "You set it up once and it runs in the background.",
        "There's a free tier and a paid plan with more features.",
        "It syncs across your devices automatically.",
        "The interface is straightforward and easy to navigate.",
        "It integrates with the tools you probably already use.",
        "Support is available if you run into questions.",
    ],
    "proof": [
        "It has solid ratings from a range of users.",
        "Plenty of {audience} use it day to day.",
        "It's been around long enough to be stable.",
    ],
    "cta": [
        "Check out {b} if it sounds useful.",
        "You can learn more about {b} on their site.",
        "Visit {b} to see the current plans.",
        "Take a look at {b} when you have a minute.",
    ],
}

NEGATIVE = {
    "hook": [
        "STOP scrolling, {b} will change your life in 7 days GUARANTEED.",
        "Doctors HATE this one weird {noun} trick.",
        "You're basically throwing money away if you ignore {b}.",
        "Everyone else is already using {b}, why aren't you?",
        "Warning: {subject} is ruining your life and {b} is the only fix.",
        "Tired of being broke and lazy? {b} is your last chance.",
    ],
    "body": [
        "Limited spots, the price doubles at midnight, act NOW.",
        "Forget everything you know, the experts are lying to you.",
        "It's only $9, then $89 a month, but trust us it's worth it.",
        "Cancel anytime (terms apply, fees may vary, no refunds).",
        "Results not typical and honestly not guaranteed but whatever.",
        "Just enter your card and all your problems disappear.",
        "If it doesn't work you probably didn't try hard enough.",
    ],
    "proof": [
        "Definitely-real customers gave it five stars, probably.",
        "Influencers we paid say it's amazing.",
        "Millions can't be wrong, click before you think too hard.",
    ],
    "cta": [
        "Smash that link before this offer is GONE forever.",
        "Don't be the only one left out, buy {b} NOW.",
        "Enter your email, phone, and card to claim your spot.",
        "Last chance, this deal will never come back (until tomorrow).",
    ],
}

TONE_POOLS = {"positive": POSITIVE, "neutral": NEUTRAL, "negative": NEGATIVE}


def make_brand(rng: random.Random) -> str:
    return rng.choice(BRAND_PREFIX) + rng.choice(BRAND_SUFFIX)


def fill(line: str, brand: str, topic: dict) -> str:
    return (
        line.replace("{b}", brand)
        .replace("{noun}", topic["noun"])
        .replace("{subject}", topic["subject"])
        .replace("{audience}", topic["audience"])
    )


def build_text(rng: random.Random, tone: str, brand: str, topic: dict) -> str:
    pool = TONE_POOLS[tone]
    parts = [
        rng.choice(pool["hook"]),
        rng.choice(pool["body"]),
    ]
    if rng.random() < 0.7:
        # second, distinct body line
        second = rng.choice(pool["body"])
        if second != parts[1]:
            parts.append(second)
    if rng.random() < 0.6:
        parts.append(rng.choice(pool["proof"]))
    parts.append(rng.choice(pool["cta"]))
    text = " ".join(fill(p, brand, topic) for p in parts)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def build_segments(text: str) -> tuple[list[dict], float]:
    sentences = split_sentences(text)
    segments: list[dict] = []
    clock = round(rng_uniform_start(), 2)
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + SENTENCES_PER_SEGMENT])
        words = max(1, len(chunk.split()))
        dur = max(1.2, words / WORDS_PER_SECOND)
        start = round(clock, 2)
        end = round(clock + dur, 2)
        segments.append({"start": start, "end": end, "text": " " + chunk})
        clock = end
        i += SENTENCES_PER_SEGMENT
    return segments, round(clock, 2)


def rng_uniform_start() -> float:
    # small natural lead-in before first word
    return random.uniform(0.0, 0.4)


def softmax(logits: dict[str, float]) -> dict[str, float]:
    mx = max(logits.values())
    exps = {k: math.exp(v - mx) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def build_reactions(rng: random.Random, tone: str, topic: dict) -> dict[str, float]:
    """Reaction probabilities conditioned on tone + a per-video quality jitter."""
    # quality latent within tone band
    if tone == "positive":
        q = rng.uniform(0.6, 1.0)
    elif tone == "neutral":
        q = rng.uniform(0.35, 0.7)
    else:
        q = rng.uniform(0.0, 0.4)

    pull = topic["emotional_pull"]
    n = lambda s=0.35: rng.gauss(0.0, s)

    logits = {
        # neutral dominates when content is unremarkable (q near 0.5)
        "neutral": 1.3 - 2.0 * abs(q - 0.5) + n(0.25),
        # likes scale with quality
        "like": -0.4 + 3.0 * q + n(),
        # dislikes scale with poorness
        "dislike": -0.4 + 3.0 * (1.0 - q) + n(),
        # positive resharing: needs strong quality, rarer, boosted by emotional pull
        "like_share": -2.1 + 3.2 * (q ** 1.5) + 0.8 * pull + n(0.4),
        # hate-sharing: rare, spikes on very bad/controversial content
        "dislike_share": -2.6 + 3.0 * ((1.0 - q) ** 1.8) + n(0.4),
    }
    probs = softmax(logits)
    return {k: round(v, 4) for k, v in probs.items()}


def build_emotions(rng: random.Random, tone: str, topic: dict,
                   reactions: dict[str, float]) -> dict[str, float]:
    pull = topic["emotional_pull"]
    positivity = reactions["like"] + reactions["like_share"]
    negativity = reactions["dislike"] + reactions["dislike_share"]

    def clamp(x: float) -> float:
        return round(min(1.0, max(0.0, x)), 4)

    empathy = clamp(0.15 + 0.75 * pull + 0.25 * positivity - 0.2 * negativity
                    + rng.gauss(0, 0.08))
    relation = clamp(0.2 + 0.55 * positivity + 0.3 * pull
                     - 0.25 * negativity + rng.gauss(0, 0.08))
    inspiration = clamp((0.1 + 0.9 * positivity) * (0.5 + 0.5 * pull)
                        + rng.gauss(0, 0.07))
    curiosity = clamp(0.35 + 0.3 * (1.0 - abs(positivity - negativity))
                      + rng.gauss(0, 0.1))
    joy = clamp(0.1 + 0.85 * positivity - 0.3 * negativity + rng.gauss(0, 0.07))

    return {
        "empathy": empathy,
        "relation": relation,
        "inspiration": inspiration,
        "curiosity": curiosity,
        "joy": joy,
    }


def make_record(index: int, rng: random.Random, seen: set[str]) -> dict:
    tone = rng.choices(
        ["positive", "neutral", "negative"], weights=[0.4, 0.35, 0.25]
    )[0]
    topic = rng.choice(TOPICS)

    # ensure a unique transcript text
    for _ in range(40):
        brand = make_brand(rng)
        text = build_text(rng, tone, brand, topic)
        if text not in seen:
            break
    seen.add(text)

    # rebuild segments with a deterministic-ish but varied start
    random.seed(rng.random())
    segments, duration = build_segments(text)
    reactions = build_reactions(rng, tone, topic)
    emotions = build_emotions(rng, tone, topic, reactions)

    rec_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return {
        "id": rec_id,
        "topic": topic["name"],
        "tone": tone,
        "language": "en",
        "language_probability": round(rng.uniform(0.985, 0.9999), 6),
        "duration": duration,
        "text": text,
        "segments": segments,
        "reactions": reactions,
        "emotions": emotions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    records: list[dict] = []
    for i in range(args.count):
        records.append(make_record(i, rng, seen))

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # quick distribution report
    tones: dict[str, int] = {}
    avg = {k: 0.0 for k in records[0]["reactions"]}
    for rec in records:
        tones[rec["tone"]] = tones.get(rec["tone"], 0) + 1
        for k, v in rec["reactions"].items():
            avg[k] += v
    n = len(records)
    print(f"Wrote {n} records to {out_path}")
    print(f"Unique transcripts: {len(seen)}")
    print(f"Tone counts: {tones}")
    print("Mean reaction probabilities:")
    for k, v in avg.items():
        print(f"  {k:14s} {v / n:.4f}")


if __name__ == "__main__":
    main()
