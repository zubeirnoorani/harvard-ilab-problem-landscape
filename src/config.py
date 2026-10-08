from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "merged_clean.csv"
    outputs: Path = PROJECT_ROOT / "outputs"
    tables: Path = PROJECT_ROOT / "outputs" / "tables"
    figures: Path = PROJECT_ROOT / "outputs" / "figures"
    cluster_examples: Path = PROJECT_ROOT / "outputs" / "cluster_examples"
    profiles: Path = PROJECT_ROOT / "outputs" / "profiles"
    artifacts: Path = PROJECT_ROOT / "artifacts"
    embedding_cache: Path = PROJECT_ROOT / "artifacts" / "embeddings"
    checkpoints: Path = PROJECT_ROOT / "artifacts" / "checkpoints"
    annotations: Path = PROJECT_ROOT / "artifacts" / "annotations"

    def ensure(self) -> None:
        for path in (
            self.tables,
            self.figures,
            self.cluster_examples,
            self.profiles,
            self.embedding_cache,
            self.checkpoints,
            self.annotations,
        ):
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()
RANDOM_SEED = 42
VALID_YEARS = (2021, 2022, 2023, 2024)
VALID_TRACKS = ("Open", "Social Impact", "Health & Life Science")
CONCEPT_GRANULARITIES = (16, 32)
K_ACTIVE = 4
STABILITY_SEEDS = (42, 7, 2024)
LOCAL_EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
INTERPRETER_MODEL = os.getenv("ILAB_INTERPRETER_MODEL", "gpt-6-luna")
API_BUDGET_USD = float(os.getenv("ILAB_API_BUDGET_USD", "5.00"))

RATING_COLUMNS = {
    "recommendation": "Recommendation",
    "problem_customer_definition": "Problem &amp; Customer Definition",
    "solution_prototype": "Prototype/MVP",
    "business_model": "Business Model",
    "impact": "Impact",
}

