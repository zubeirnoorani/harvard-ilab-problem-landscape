from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version

from .aggregate_startups import build_startup_level, write_startup_outputs
from .analysis import run_analysis, write_friday_findings
from .build_problem_text import build_problem_text_dataset, write_problem_text_outputs
from .clustering import (
    UPSTREAM_COMMIT,
    UPSTREAM_ROOT,
    apply_curated_labels,
    compare_granularities,
    embed_problem_texts,
    train_concept_landscape,
    stability_diagnostics,
)
from .config import CONCEPT_GRANULARITIES, PATHS
from .load_data import create_audit_outputs, load_raw_data, validate_core_shape
from .visualization import generate_all_visualizations


def ensure_upstream() -> None:
    if not (UPSTREAM_ROOT / ".git").exists():
        UPSTREAM_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/rmovva/HypotheSAEs.git", str(UPSTREAM_ROOT)],
            check=True,
        )
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM_ROOT, text=True).strip()
    if current != UPSTREAM_COMMIT:
        subprocess.run(["git", "checkout", UPSTREAM_COMMIT], cwd=UPSTREAM_ROOT, check=True)


def write_run_manifest(problem_df, judge_rows: int) -> None:
    raw_hash = hashlib.sha256(PATHS.raw_data.read_bytes()).hexdigest()
    packages = {}
    for package in ("pandas", "numpy", "scikit-learn", "torch", "sentence-transformers", "streamlit"):
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = "unknown"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": raw_hash,
        "judge_rows": int(judge_rows),
        "applications": int(problem_df["application_id"].nunique()),
        "eligible_problem_texts": int((~problem_df["problem_text_missing"]).sum()),
        "hypothesaes_commit": UPSTREAM_COMMIT,
        "embedding_model": "nomic-ai/modernbert-embed-base",
        "sae": {"m": [16, 32], "k": 4, "seed": 42},
        "taxonomy_uses_judging_outcomes": False,
        "raw_data_committed": False,
    }
    (PATHS.tables / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_pipeline(force_embeddings: bool = False, force_sae: bool = False, epochs: int = 500) -> None:
    PATHS.ensure()
    raw = load_raw_data()
    validate_core_shape(raw)
    create_audit_outputs(raw)

    startup, judge, reconciliation = build_startup_level(raw)
    write_startup_outputs(startup, judge, reconciliation)
    problem_df = build_problem_text_dataset(startup)
    write_problem_text_outputs(problem_df)

    ensure_upstream()
    eligible, embeddings, _ = embed_problem_texts(problem_df, force=force_embeddings)
    results = {}
    for m_concepts in CONCEPT_GRANULARITIES:
        results[m_concepts] = train_concept_landscape(
            problem_df,
            m_concepts,
            embeddings=embeddings,
            eligible=eligible,
            n_epochs=epochs,
            force=force_sae,
        )
        diagnostics, assignments = apply_curated_labels(m_concepts)
        results[m_concepts]["diagnostics"] = diagnostics
        results[m_concepts]["assignments"] = assignments
        stability_diagnostics(
            embeddings,
            m_concepts,
            results[m_concepts]["activations"],
            n_epochs=epochs,
        )
    compare_granularities(results)

    for m_concepts in CONCEPT_GRANULARITIES:
        run_analysis(m_concepts)
        generate_all_visualizations(m_concepts)
    write_friday_findings(16)
    write_run_manifest(problem_df, len(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Harvard i-lab problem landscape prototype.")
    parser.add_argument("--force-embeddings", action="store_true", help="Recompute local embeddings.")
    parser.add_argument("--force-sae", action="store_true", help="Retrain SAE artifacts.")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum SAE epochs.")
    args = parser.parse_args()
    run_pipeline(args.force_embeddings, args.force_sae, args.epochs)


if __name__ == "__main__":
    main()
