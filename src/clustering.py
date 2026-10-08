from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

from .config import K_ACTIVE, LOCAL_EMBEDDING_MODEL, PATHS, RANDOM_SEED
from .concept_labels import CURATED_LABELS


UPSTREAM_ROOT = PATHS.root / "external" / "HypotheSAEs"
UPSTREAM_COMMIT = "706d8979c71befe8e3151f51a8d4dbf87d4d7e41"


def _load_upstream():
    """Import the pinned, unmodified HypotheSAEs checkout."""
    if not (UPSTREAM_ROOT / "hypothesaes" / "sae.py").exists():
        raise FileNotFoundError(
            "Missing external/HypotheSAEs. Run `git clone https://github.com/rmovva/"
            "HypotheSAEs.git external/HypotheSAEs` and checkout " + UPSTREAM_COMMIT
        )
    os.environ.setdefault("EMB_CACHE_DIR", str(PATHS.embedding_cache / "upstream"))
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    from hypothesaes.embedding import get_local_embeddings
    from hypothesaes.sae import SparseAutoencoder, load_model

    return get_local_embeddings, SparseAutoencoder, load_model


def _corpus_fingerprint(application_ids: list[str], texts: list[str], model: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode())
    for app_id, text in zip(application_ids, texts):
        digest.update(app_id.encode())
        digest.update(b"\0")
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def embed_problem_texts(
    problem_df: pd.DataFrame,
    model_name: str = LOCAL_EMBEDDING_MODEL,
    force: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    eligible = problem_df.loc[~problem_df["problem_text_missing"]].copy().reset_index(drop=True)
    texts = eligible["problem_text"].tolist()
    ids = eligible["application_id"].astype(str).tolist()
    fingerprint = _corpus_fingerprint(ids, texts, model_name)
    safe_model = model_name.replace("/", "__")
    ordered_cache = PATHS.embedding_cache / f"problem_text_{safe_model}_{fingerprint}.npz"
    if ordered_cache.exists() and not force:
        cached = np.load(ordered_cache)
        cached_ids = cached["application_ids"].astype(str).tolist()
        if cached_ids == ids:
            embeddings = cached["embeddings"].astype(np.float32)
            return eligible, embeddings, {
                "model": model_name,
                "fingerprint": fingerprint,
                "cache": str(ordered_cache),
                "n_texts": len(texts),
                "dimension": embeddings.shape[1],
            }

    get_local_embeddings, _, _ = _load_upstream()
    mapping = get_local_embeddings(
        texts,
        model=model_name,
        batch_size=32,
        cache_name=f"problem_text_{safe_model}_{fingerprint}",
        show_progress=True,
    )
    embeddings = np.stack([mapping[text] for text in texts]).astype(np.float32)
    embeddings = normalize(embeddings, norm="l2").astype(np.float32)
    ordered_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ordered_cache,
        application_ids=np.asarray(ids, dtype=str),
        embeddings=embeddings,
    )
    metadata = {
        "model": model_name,
        "fingerprint": fingerprint,
        "cache": str(ordered_cache),
        "n_texts": len(texts),
        "dimension": int(embeddings.shape[1]),
    }
    (PATHS.embedding_cache / "problem_text_embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return eligible, embeddings, metadata


def _distinctive_terms(texts: list[str], activations: np.ndarray, n_terms: int = 4) -> list[list[str]]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=20_000,
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    baseline = np.asarray(matrix.mean(axis=0)).ravel()
    banned = {
        "customer", "customers", "stakeholder", "stakeholders", "problem", "problems",
        "need", "needs", "people", "currently", "lack", "access", "including",
    }
    labels: list[list[str]] = []
    for concept in range(activations.shape[1]):
        weights = activations[:, concept].copy()
        if weights.sum() == 0:
            labels.append([f"inactive feature {concept:02d}"])
            continue
        weighted = np.asarray(matrix.multiply(weights[:, None]).sum(axis=0)).ravel() / weights.sum()
        contrast = weighted - 0.35 * baseline
        ranked = np.argsort(contrast)[::-1]
        chosen = []
        for idx in ranked:
            term = str(terms[idx])
            tokens = set(term.split())
            if tokens.issubset(banned) or any(term in prior or prior in term for prior in chosen):
                continue
            chosen.append(term)
            if len(chosen) == n_terms:
                break
        labels.append(chosen)
    return labels


def _example_payload(df: pd.DataFrame, indices: np.ndarray, activation: np.ndarray) -> str:
    examples = []
    for idx in indices:
        row = df.iloc[int(idx)]
        examples.append(
            {
                "application_id": str(row["application_id"]),
                "venture": str(row["venture_name"]),
                "year": int(row["year"]),
                "track": str(row["Track"]),
                "activation": round(float(activation[int(idx)]), 4),
                "problem_text": str(row["problem_text"])[:900],
            }
        )
    return json.dumps(examples, ensure_ascii=False)


def _concept_diagnostics(
    eligible: pd.DataFrame,
    embeddings: np.ndarray,
    activations: np.ndarray,
    m_concepts: int,
    reconstruction_nrmse: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    term_sets = _distinctive_terms(eligible["problem_text"].tolist(), activations)
    rows = []
    for concept in range(m_concepts):
        values = activations[:, concept]
        active = np.flatnonzero(values > 0)
        ranked = active[np.argsort(values[active])[::-1]] if len(active) else active
        top = ranked[:5]
        if len(ranked) <= 5:
            moderate = ranked
        else:
            lo = max(5, int(len(ranked) * 0.35))
            hi = max(lo + 1, int(len(ranked) * 0.65))
            moderate = ranked[lo:hi][:5]
        terms = term_sets[concept]
        label = " / ".join(term.title() for term in terms[:3])
        n_active = int(len(active))
        notes = []
        if n_active < 10:
            notes.append("small-N; interpret cautiously")
        if n_active == 0:
            notes.append("inactive/dead feature")
        rows.append(
            {
                "concept_id": concept,
                "concept_label": label,
                "description": f"Problem statements distinguished by: {', '.join(terms)}.",
                "prevalence": n_active / len(eligible),
                "n_startups": n_active,
                "mean_activation_when_active": float(values[active].mean()) if n_active else np.nan,
                "top_examples": _example_payload(eligible, top, values),
                "moderate_examples": _example_payload(eligible, moderate, values),
                "problem_quality_notes": "; ".join(notes) or "manual demand-side review required",
                "label_status": "provisional_distinctive_terms",
                "embedding_model": LOCAL_EMBEDDING_MODEL,
                "sae_m": m_concepts,
                "sae_k": K_ACTIVE,
                "reconstruction_nrmse": reconstruction_nrmse,
            }
        )
    diagnostics = pd.DataFrame(rows)

    activation_cols = [f"concept_{i:02d}_activation" for i in range(m_concepts)]
    assignment = eligible[
        [
            "application_id", "year", "Track", "venture_name", "problem_text",
            "problem_text_quality", "School", "Gender", "venture location country",
            "Industries", "industry primary",
        ]
    ].copy()
    assignment[activation_cols] = activations
    primary = np.argmax(activations, axis=1)
    max_value = activations[np.arange(len(activations)), primary]
    primary = np.where(max_value > 0, primary, -1)
    label_map = diagnostics.set_index("concept_id")["concept_label"].to_dict()
    assignment["primary_problem_concept"] = primary
    assignment["primary_problem_label"] = [label_map.get(int(value), "Unassigned") for value in primary]
    assignment["primary_activation"] = max_value

    overlap_rows = []
    memberships = activations > 0
    for left in range(m_concepts):
        for right in range(left + 1, m_concepts):
            union = np.logical_or(memberships[:, left], memberships[:, right]).sum()
            intersection = np.logical_and(memberships[:, left], memberships[:, right]).sum()
            overlap_rows.append(
                {
                    "concept_a": left,
                    "concept_b": right,
                    "n_overlap": int(intersection),
                    "jaccard": float(intersection / union) if union else 0.0,
                    "activation_correlation": float(np.corrcoef(activations[:, left], activations[:, right])[0, 1])
                    if np.std(activations[:, left]) and np.std(activations[:, right])
                    else np.nan,
                }
            )
    return diagnostics, assignment, pd.DataFrame(overlap_rows)


def train_concept_landscape(
    problem_df: pd.DataFrame,
    m_concepts: int,
    embeddings: np.ndarray | None = None,
    eligible: pd.DataFrame | None = None,
    seed: int = RANDOM_SEED,
    n_epochs: int = 500,
    force: bool = False,
) -> dict[str, object]:
    import torch

    _, SparseAutoencoder, load_model = _load_upstream()
    if eligible is None or embeddings is None:
        eligible, embeddings, _ = embed_problem_texts(problem_df)
    checkpoint = PATHS.checkpoints / f"SAE_M={m_concepts}_K={K_ACTIVE}_seed={seed}.pt"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if checkpoint.exists() and not force:
        model = load_model(str(checkpoint), device="cpu")
        history = {"loaded_from_cache": True}
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(embeddings))
        split = max(1, int(len(order) * 0.85))
        train_idx, val_idx = order[:split], order[split:]
        model = SparseAutoencoder(
            input_dim=embeddings.shape[1],
            m_total_neurons=m_concepts,
            k_active_neurons=K_ACTIVE,
            aux_k=min(2 * K_ACTIVE, m_concepts),
            dead_neuron_threshold_steps=100,
            device="cpu",
        )
        history = model.fit(
            torch.from_numpy(embeddings[train_idx]).float(),
            torch.from_numpy(embeddings[val_idx]).float(),
            batch_size=min(128, len(train_idx)),
            learning_rate=1e-3,
            n_epochs=n_epochs,
            patience=35,
            show_progress=True,
        )
        model.save(str(checkpoint))

    activations = model.get_activations(embeddings, show_progress=False)
    with torch.no_grad():
        reconstructed, _ = model(torch.from_numpy(embeddings).float())
    residual = np.mean((reconstructed.cpu().numpy() - embeddings) ** 2)
    baseline = np.mean((embeddings - embeddings.mean(axis=0, keepdims=True)) ** 2)
    reconstruction_nrmse = float(residual / baseline)
    diagnostics, assignments, overlap = _concept_diagnostics(
        eligible, embeddings, activations, m_concepts, reconstruction_nrmse
    )
    diagnostics.to_csv(PATHS.tables / f"concept_diagnostics_m{m_concepts}.csv", index=False)
    assignments.to_csv(PATHS.tables / f"concept_assignments_m{m_concepts}.csv", index=False)
    overlap.to_csv(PATHS.tables / f"concept_overlap_m{m_concepts}.csv", index=False)
    np.save(PATHS.annotations / f"activations_m{m_concepts}.npy", activations)
    history_path = PATHS.checkpoints / f"history_m{m_concepts}_seed={seed}.json"
    serializable = {
        key: [float(x) for x in value] if isinstance(value, list) else value
        for key, value in history.items()
    }
    history_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return {
        "diagnostics": diagnostics,
        "assignments": assignments,
        "overlap": overlap,
        "activations": activations,
        "embeddings": embeddings,
        "eligible": eligible,
        "history": history,
    }


def compare_granularities(results: dict[int, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for m_concepts, result in results.items():
        assignments = result["assignments"]
        diagnostics = result["diagnostics"]
        labels = assignments["primary_problem_concept"].to_numpy()
        valid = labels >= 0
        silhouette = np.nan
        if valid.sum() > len(np.unique(labels[valid])) > 1:
            silhouette = silhouette_score(result["embeddings"][valid], labels[valid], metric="cosine")
        primary_counts = assignments.loc[valid, "primary_problem_concept"].value_counts()
        rows.append(
            {
                "m_concepts": m_concepts,
                "k_active": K_ACTIVE,
                "n_eligible_startups": len(assignments),
                "n_active_features": int((diagnostics["n_startups"] > 0).sum()),
                "n_features_under_10_members": int((diagnostics["n_startups"] < 10).sum()),
                "median_memberships_per_feature": float(diagnostics["n_startups"].median()),
                "median_primary_area_size": float(primary_counts.median()),
                "minimum_primary_area_size": int(primary_counts.min()),
                "cosine_silhouette_primary_assignment": silhouette,
                "reconstruction_nrmse": float(diagnostics["reconstruction_nrmse"].iloc[0]),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("m_concepts")
    comparison.to_csv(PATHS.tables / "concept_granularity_comparison.csv", index=False)
    return comparison


def apply_curated_labels(m_concepts: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply labels created by manual review of top and moderate activations."""
    labels = CURATED_LABELS[m_concepts]
    diagnostics_path = PATHS.tables / f"concept_diagnostics_m{m_concepts}.csv"
    assignments_path = PATHS.tables / f"concept_assignments_m{m_concepts}.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    assignments = pd.read_csv(assignments_path, low_memory=False)
    if set(diagnostics["concept_id"].astype(int)) != set(labels):
        raise ValueError(f"Curated label mapping does not match M={m_concepts} concepts")
    diagnostics["concept_label"] = diagnostics["concept_id"].map(lambda x: labels[int(x)]["label"])
    diagnostics["description"] = diagnostics["concept_id"].map(lambda x: labels[int(x)]["description"])
    diagnostics["problem_quality_notes"] = diagnostics["concept_id"].map(lambda x: labels[int(x)]["notes"])
    diagnostics["label_status"] = "human_reviewed_top_and_moderate_examples"
    label_map = diagnostics.set_index("concept_id")["concept_label"].to_dict()
    assignments["primary_problem_label"] = assignments["primary_problem_concept"].map(label_map).fillna("Unassigned")
    diagnostics.to_csv(diagnostics_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    for _, row in diagnostics.iterrows():
        lines = [
            f"# M={m_concepts} · Concept {int(row['concept_id']):02d}: {row['concept_label']}",
            "",
            str(row["description"]),
            "",
            f"Prevalence: {row['prevalence']:.1%} ({int(row['n_startups'])} active memberships)",
            "",
            f"Review note: {row['problem_quality_notes']}",
            "",
            "## Strongest activations",
            "",
        ]
        for example in json.loads(row["top_examples"]):
            lines.extend(
                [
                    f"### {example['venture']} ({example['year']}, {example['track']})",
                    "",
                    str(example["problem_text"]),
                    "",
                ]
            )
        lines.extend(["## Moderate activations", ""])
        for example in json.loads(row["moderate_examples"]):
            lines.extend(
                [
                    f"### {example['venture']} ({example['year']}, {example['track']})",
                    "",
                    str(example["problem_text"]),
                    "",
                ]
            )
        (PATHS.cluster_examples / f"concept_m{m_concepts}_{int(row['concept_id']):02d}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return diagnostics, assignments


def _activations_for_seed(
    embeddings: np.ndarray,
    m_concepts: int,
    seed: int,
    n_epochs: int = 500,
) -> np.ndarray:
    import torch

    _, SparseAutoencoder, load_model = _load_upstream()
    checkpoint = PATHS.checkpoints / f"SAE_M={m_concepts}_K={K_ACTIVE}_seed={seed}.pt"
    if checkpoint.exists():
        model = load_model(str(checkpoint), device="cpu")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        order = np.random.default_rng(seed).permutation(len(embeddings))
        split = max(1, int(len(order) * 0.85))
        model = SparseAutoencoder(
            input_dim=embeddings.shape[1],
            m_total_neurons=m_concepts,
            k_active_neurons=K_ACTIVE,
            aux_k=min(2 * K_ACTIVE, m_concepts),
            dead_neuron_threshold_steps=100,
            device="cpu",
        )
        model.fit(
            torch.from_numpy(embeddings[order[:split]]).float(),
            torch.from_numpy(embeddings[order[split:]]).float(),
            batch_size=min(128, split),
            learning_rate=1e-3,
            n_epochs=n_epochs,
            patience=35,
            show_progress=False,
        )
        model.save(str(checkpoint))
    return model.get_activations(embeddings, show_progress=False)


def stability_diagnostics(
    embeddings: np.ndarray,
    m_concepts: int,
    main_activations: np.ndarray,
    seeds: tuple[int, ...] = (7, 2024),
    n_epochs: int = 500,
) -> pd.DataFrame:
    """Align alternate-seed features to seed 42 and quantify activation stability."""
    diagnostics = pd.read_csv(PATHS.tables / f"concept_diagnostics_m{m_concepts}.csv")
    assignments = pd.read_csv(PATHS.tables / f"concept_assignments_m{m_concepts}.csv")
    primary_sizes = assignments["primary_problem_concept"].value_counts().to_dict()
    result = diagnostics[["concept_id", "concept_label", "n_startups"]].copy()
    correlation_columns = []
    jaccard_columns = []
    for seed in seeds:
        alternate = _activations_for_seed(embeddings, m_concepts, seed, n_epochs=n_epochs)
        corr = np.full((m_concepts, m_concepts), -1.0)
        for left in range(m_concepts):
            for right in range(m_concepts):
                if np.std(main_activations[:, left]) and np.std(alternate[:, right]):
                    corr[left, right] = np.corrcoef(main_activations[:, left], alternate[:, right])[0, 1]
        left_idx, right_idx = linear_sum_assignment(-corr)
        match = dict(zip(left_idx, right_idx))
        corr_col = f"matched_activation_correlation_seed_{seed}"
        jaccard_col = f"matched_membership_jaccard_seed_{seed}"
        result[corr_col] = [float(corr[concept, match[concept]]) for concept in result["concept_id"]]
        jaccards = []
        for concept in result["concept_id"]:
            left_members = main_activations[:, int(concept)] > 0
            right_members = alternate[:, match[int(concept)]] > 0
            union = np.logical_or(left_members, right_members).sum()
            jaccards.append(float(np.logical_and(left_members, right_members).sum() / union) if union else 0.0)
        result[jaccard_col] = jaccards
        correlation_columns.append(corr_col)
        jaccard_columns.append(jaccard_col)
    result["mean_matched_activation_correlation"] = result[correlation_columns].mean(axis=1)
    result["mean_matched_membership_jaccard"] = result[jaccard_columns].mean(axis=1)
    result["n_primary_assignments"] = result["concept_id"].map(primary_sizes).fillna(0).astype(int)
    result["small_primary_area"] = result["n_primary_assignments"] < 10
    result["stability_flag"] = np.select(
        [
            result["mean_matched_activation_correlation"] < 0.35,
            result["mean_matched_activation_correlation"] < 0.55,
        ],
        ["low", "moderate"],
        default="higher",
    )
    result["small_and_unstable"] = result["small_primary_area"] & result["stability_flag"].isin(["low", "moderate"])
    result["m_concepts"] = m_concepts
    result.to_csv(PATHS.tables / f"concept_stability_m{m_concepts}.csv", index=False)
    return result
