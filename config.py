"""Configuration for golden dataset generation."""

from pathlib import Path

# --- Paths ---
DOCS_ROOT: Path = Path("data/raw/docs")
OUTPUT_DIR: Path = Path("output")
OUTPUT_FILE: Path = OUTPUT_DIR / "golden_dataset.json"

# --- OpenAI Model ---
MODEL_NAME: str = "gpt-5.4-2026-03-05"
REASONING_EFFORT: str = "high"

# --- Dataset Targets ---
TARGET_QA_PAIRS: int = 60  # aim for middle of 50-100 range
MAX_TOKENS_PER_CALL: int = 16000

# --- Question Type Distribution (target percentages) ---
QUESTION_TYPE_TARGETS: dict[str, float] = {
    "factual": 0.18,
    "conceptual": 0.18,
    "procedural": 0.18,
    "comparative": 0.15,
    "multi_hop": 0.16,
    "edge_case": 0.15,
}

# --- Document Loading ---
SUPPORTED_EXTENSIONS: list[str] = [".md", ".mdx"]
MIN_DOC_LENGTH_CHARS: int = 200  # skip trivially small files
