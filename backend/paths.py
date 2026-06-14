"""Central path constants for the milphacks project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PERSONAS_DIR = DATA_DIR / "personas"
VIDEOS_DIR = DATA_DIR / "videos"
TRANSCRIPTS_DIR = ROOT / "transcripts"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
SIMULATIONS_DIR = OUTPUTS_DIR / "simulations"
TESTS_DIR = ROOT / "tests"

PERSONA_CSV = PERSONAS_DIR / "nemotron_sample_10k.csv"
PERSONA_EMBEDDINGS_CSV = PERSONAS_DIR / "nemotron_sample_10k_embeddings_noname.csv"
PERSONA_EMBEDDINGS_WEIGHTED_CSV = (
    PERSONAS_DIR / "nemotron_sample_10k_embeddings_noname_weighted.csv"
)
PERSONA_EMBEDDINGS_MINMAX_CSV = (
    PERSONAS_DIR / "nemotron_sample_10k_embeddings_noname_weighted_minmax.csv"
)

TRAINING_JSONL = TRANSCRIPTS_DIR / "training_data.jsonl"
TRAINING_EMBEDDED_JSONL = TRANSCRIPTS_DIR / "training_data_with_embeddings.jsonl"
TRAINING_SEGMENT_EMBEDDINGS_CSV = (
    TRANSCRIPTS_DIR / "training_data_segment_embeddings.csv"
)

# Aliases used by embedding scripts.
DEFAULT_INPUT = TRAINING_JSONL
DEFAULT_OUTPUT = TRAINING_EMBEDDED_JSONL
DEFAULT_CSV_OUTPUT = TRAINING_SEGMENT_EMBEDDINGS_CSV

PERSONA_WEIGHTS = MODELS_DIR / "persona_pathpool.pt"
PATHPOOL_WEIGHTS = MODELS_DIR / "pathpool_classifier.pt"
DEFAULT_SIM_OUTPUT = SIMULATIONS_DIR / "network_simulation.json"

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SHARPNESS = 2.5


def rel(path: str | Path) -> str:
    """Return a repo-relative path string for portable exports."""
    return str(Path(path).resolve().relative_to(ROOT))
