"""Build the disclosure-controlled payload used by the hosted research atlas.

Only aggregate cells with at least ``MIN_CELL`` observations are published.
Application text, venture names, identifiers, row-level demographics, embeddings,
activations, checkpoints, and example excerpts never enter the hosted payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
OUTPUT = ROOT / "vercel-site" / "data" / "landscape.json"
MIN_CELL = 10
YEARS = [2021, 2022, 2023, 2024]
SCOPE_MAP: dict[str, tuple[str, str | None, str]] = {
    "All tracks": ("All tracks", None, "All tracks"),
    "Open": ("Track", "Open", "Open"),
    "Social Impact": ("Track", "Social Impact", "Social Impact"),
    "Health & Life Sciences": ("Track", "Health & Life Science", "Health & Life Science"),
}


def clean_number(value: object, digits: int = 4) -> float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def clean_label(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Not reported"
    label = str(value).strip()
    if label == "Health & Life Science":
        return "Health & Life Sciences"
    return label


def outcome_record(row: pd.Series, trend: pd.Series | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": int(row["concept_id"]),
        "label": str(row["concept_label"]),
        "n": int(row["n_startups"]),
        "scopeN": int(row["scope_n_startups"]),
        "share": clean_number(row["share_of_scope"]),
        "recommendation": clean_number(row["recommendation_mean"]),
        "recommendationSd": clean_number(row["recommendation_sd_between_startups"]),
        "recommendationMedian": clean_number(row["recommendation_median"]),
        "recommendationStartups": clean_number(row["recommendation_n_startups"]),
        "ratings": clean_number(row["recommendation_n_ratings"]),
        "problem": clean_number(row["problem_customer_definition_mean"]),
        "problemSd": clean_number(row["problem_customer_definition_sd_between_startups"]),
        "solution": clean_number(row["solution_prototype_mean"]),
        "solutionSd": clean_number(row["solution_prototype_sd_between_startups"]),
        "business": clean_number(row["business_model_mean"]),
        "businessSd": clean_number(row["business_model_sd_between_startups"]),
        "impact": clean_number(row["impact_mean"]),
        "impactSd": clean_number(row["impact_sd_between_startups"]),
        "trend": None,
        "change": None,
    }
    if trend is not None:
        record["trend"] = clean_number(trend["trend_slope_share_per_year"])
        record["change"] = clean_number(trend["change_2021_to_2024"])
    return record


def build_slices(outcomes: pd.DataFrame, trends: pd.DataFrame, membership: str) -> dict[str, Any]:
    membership_type = "primary_assignment" if membership == "primary" else "active_feature"
    base = outcomes[outcomes["membership_type"] == membership_type]
    result: dict[str, Any] = {}
    for display_scope, (scope, track, trend_scope) in SCOPE_MAP.items():
        scoped = base[base["scope"] == scope]
        if track is not None:
            scoped = scoped[scoped["Track"] == track]
        trend_lookup = trends[trends["track_scope"] == trend_scope].set_index("concept_id")
        periods: dict[str, Any] = {}
        for period in ["All years", *[str(year) for year in YEARS]]:
            if period == "All years":
                rows = scoped[scoped["period"] == "All years"]
            else:
                rows = scoped[(scoped["period"] == "Year") & (scoped["year"] == int(period))]
            scope_n = int(rows["scope_n_startups"].iloc[0]) if not rows.empty else 0
            safe = rows[rows["n_startups"] >= MIN_CELL].copy()
            items: list[dict[str, Any]] = []
            for _, row in safe.sort_values("n_startups", ascending=False).iterrows():
                trend = None
                cid = int(row["concept_id"])
                if membership == "primary" and period == "All years" and cid in trend_lookup.index:
                    trend = trend_lookup.loc[cid]
                items.append(outcome_record(row, trend))
            visible_n = sum(item["n"] for item in items)
            periods[period] = {
                "scopeN": scope_n,
                "visibleN": visible_n,
                "suppressedN": max(0, scope_n - visible_n) if membership == "primary" else None,
                "items": items,
            }
        result[display_scope] = periods
    return result


def safe_composition(
    data: pd.DataFrame,
    concept_id: int,
    category_col: str,
    count_col: str,
) -> dict[str, Any]:
    rows = data[data["concept_id"] == concept_id].copy()
    rows[category_col] = rows[category_col].map(clean_label)
    safe = rows[rows[count_col] >= MIN_CELL].sort_values(count_col, ascending=False)
    shown_total = int(safe[count_col].sum())
    values = [
        {
            "label": str(row[category_col]),
            "n": int(row[count_col]),
            "shareDisplayed": clean_number(row[count_col] / shown_total) if shown_total else None,
        }
        for _, row in safe.iterrows()
    ]
    return {
        "values": values,
        "shownN": shown_total,
        "suppressedCells": int((rows[count_col] < MIN_CELL).sum()),
    }


def build_profiles(m: int, concepts: pd.DataFrame, slices: dict[str, Any]) -> dict[str, Any]:
    compositions = {
        "tracks": (pd.read_csv(TABLES / f"composition_Track_m{m}.csv"), "Track", "n_startups"),
        "years": (pd.read_csv(TABLES / f"composition_year_m{m}.csv"), "year", "n_startups"),
        "gender": (pd.read_csv(TABLES / f"composition_gender_m{m}.csv"), "gender", "n_founders"),
        "schools": (
            pd.read_csv(TABLES / f"composition_harvard_school_m{m}.csv"),
            "harvard_school",
            "n_founders",
        ),
        "countries": (pd.read_csv(TABLES / f"composition_country_m{m}.csv"), "country", "n_founders"),
        "industries": (
            pd.read_csv(TABLES / f"composition_industry_primary_m{m}.csv"),
            "industry primary",
            "n_startups",
        ),
    }
    overlap = pd.read_csv(TABLES / f"concept_overlap_m{m}.csv")
    judge_types = pd.read_csv(TABLES / f"judge_type_m{m}.csv")
    intervention = pd.read_csv(TABLES / f"intervention_profiles_m{m}.csv").set_index("concept_id")
    trends = pd.read_csv(TABLES / f"trends_m{m}.csv")
    label_map = concepts.set_index("id")["label"].to_dict()
    primary_all = {item["id"]: item for item in slices["primary"]["All tracks"]["All years"]["items"]}
    active_all = {item["id"]: item for item in slices["active"]["All tracks"]["All years"]["items"]}

    profiles: dict[str, Any] = {}
    for concept in concepts.to_dict("records"):
        cid = int(concept["id"])
        related_rows = overlap[
            ((overlap["concept_a"] == cid) | (overlap["concept_b"] == cid))
            & (overlap["n_overlap"] >= MIN_CELL)
        ].copy()
        related_rows["other"] = np.where(
            related_rows["concept_a"] == cid,
            related_rows["concept_b"],
            related_rows["concept_a"],
        )
        related = [
            {
                "id": int(row["other"]),
                "label": label_map.get(int(row["other"]), f"Concept {int(row['other'])}"),
                "nOverlap": int(row["n_overlap"]),
                "jaccard": clean_number(row["jaccard"]),
                "activationCorrelation": clean_number(row["activation_correlation"]),
            }
            for _, row in related_rows.sort_values(["jaccard", "n_overlap"], ascending=False).head(6).iterrows()
        ]

        jt = judge_types[
            (judge_types["concept_id"] == cid)
            & (judge_types["n_startups"] >= MIN_CELL)
            & (judge_types["n_ratings"] >= MIN_CELL)
            & judge_types["judge_type"].notna()
        ]
        judge_type_records = [
            {
                "label": str(row["judge_type"]),
                "mean": clean_number(row["recommendation_mean"]),
                "sd": clean_number(row["recommendation_sd"]),
                "median": clean_number(row["recommendation_median"]),
                "ratings": int(row["n_ratings"]),
                "startups": int(row["n_startups"]),
                "adjustedMean": clean_number(row["mean_year_track_centered_rating"]),
            }
            for _, row in jt.sort_values("recommendation_mean", ascending=False).iterrows()
        ]

        primary = primary_all.get(cid)
        support: dict[str, Any] | None = None
        if cid in intervention.index and int(intervention.loc[cid, "n_startups"]) >= MIN_CELL:
            row = intervention.loc[cid]
            support = {
                "pattern": str(row["support_pattern"]),
                "n": int(row["n_startups"]),
                "problemZ": clean_number(row["problem_customer_definition_z"]),
                "solutionZ": clean_number(row["solution_prototype_z"]),
                "businessZ": clean_number(row["business_model_z"]),
                "impactZ": clean_number(row["impact_z"]),
                "problemMinusBusiness": clean_number(row["problem_minus_business"]),
                "prototypeMinusProblem": clean_number(row["prototype_minus_problem"]),
                "impactMinusBusiness": clean_number(row["impact_minus_business"]),
            }

        trend_row = trends[(trends["track_scope"] == "All tracks") & (trends["concept_id"] == cid)]
        trend_record: dict[str, Any] | None = None
        if primary is not None and not trend_row.empty:
            row = trend_row.iloc[0]
            year_values = []
            for year in YEARS:
                count = int(row[f"count_{year}"])
                year_values.append(
                    {
                        "year": year,
                        "n": count if count >= MIN_CELL else None,
                        "share": clean_number(row[f"share_{year}"]) if count >= MIN_CELL else None,
                        "suppressed": count < MIN_CELL,
                    }
                )
            trend_record = {
                "slope": clean_number(row["trend_slope_share_per_year"]),
                "change": clean_number(row["change_2021_to_2024"]),
                "years": year_values,
            }

        profiles[str(cid)] = {
            "primaryOutcome": primary,
            "activeOutcome": active_all.get(cid),
            "support": support,
            "trend": trend_record,
            "compositions": {
                name: safe_composition(frame, cid, category, count)
                for name, (frame, category, count) in compositions.items()
            },
            "judgeTypes": judge_type_records,
            "related": related,
        }
    return profiles


def build_concepts(m: int) -> pd.DataFrame:
    diagnostics = pd.read_csv(TABLES / f"concept_diagnostics_m{m}.csv")
    stability = pd.read_csv(TABLES / f"concept_stability_m{m}.csv").set_index("concept_id")
    rows: list[dict[str, Any]] = []
    for _, row in diagnostics.sort_values("concept_id").iterrows():
        cid = int(row["concept_id"])
        stable = stability.loc[cid]
        primary_n = int(stable["n_primary_assignments"])
        rows.append(
            {
                "id": cid,
                "label": str(row["concept_label"]),
                "description": str(row["description"]),
                "activeN": int(row["n_startups"]),
                "activePrevalence": clean_number(row["prevalence"]),
                "meanActivation": clean_number(row["mean_activation_when_active"]),
                "primaryN": primary_n if primary_n >= MIN_CELL else None,
                "primarySuppressed": primary_n < MIN_CELL,
                "qualityNote": str(row["problem_quality_notes"]),
                "labelStatus": str(row["label_status"]),
                "stability": str(stable["stability_flag"]),
                "activationCorrelation": clean_number(stable["mean_matched_activation_correlation"]),
                "membershipJaccard": clean_number(stable["mean_matched_membership_jaccard"]),
                "smallPrimary": bool(stable["small_primary_area"]),
                "smallAndUnstable": bool(stable["small_and_unstable"]),
            }
        )
    return pd.DataFrame(rows)


def build_founder_matrix(m: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    specs = {
        "gender": ("composition_gender", "gender"),
        "schools": ("composition_harvard_school", "harvard_school"),
        "countries": ("composition_country", "country"),
    }
    for name, (file_stem, category) in specs.items():
        data = pd.read_csv(TABLES / f"{file_stem}_m{m}.csv")
        safe = data[data["n_founders"] >= MIN_CELL].copy()
        result[name] = [
            {
                "conceptId": int(row["concept_id"]),
                "conceptLabel": str(row["concept_label"]),
                "category": clean_label(row[category]),
                "n": int(row["n_founders"]),
            }
            for _, row in safe.iterrows()
        ]
    industry = pd.read_csv(TABLES / f"composition_industry_primary_m{m}.csv")
    industry = industry[industry["n_startups"] >= MIN_CELL]
    result["industries"] = [
        {
            "conceptId": int(row["concept_id"]),
            "conceptLabel": str(row["concept_label"]),
            "category": clean_label(row["industry primary"]),
            "n": int(row["n_startups"]),
        }
        for _, row in industry.iterrows()
    ]
    return result


def build_disagreement(m: int) -> dict[str, Any]:
    data = pd.read_csv(TABLES / f"judge_disagreement_m{m}.csv")
    result: dict[str, Any] = {}
    for display_scope, (_, _, raw_scope) in SCOPE_MAP.items():
        rows = data[(data["track_scope"] == raw_scope) & (data["n_startups"] >= MIN_CELL)]
        result[display_scope] = [
            {
                "id": int(row["concept_id"]),
                "label": str(row["concept_label"]),
                "recommendation": clean_number(row["recommendation_mean"]),
                "meanWithinStartupSd": clean_number(row["mean_within_startup_recommendation_sd"]),
                "medianWithinStartupSd": clean_number(row["median_within_startup_recommendation_sd"]),
                "n": int(row["n_startups"]),
                "ratings": int(row["n_underlying_ratings"]),
                "quadrant": str(row["rating_disagreement_quadrant"]),
            }
            for _, row in rows.iterrows()
        ]
    return result


def build_model(m: int, summary: pd.Series) -> dict[str, Any]:
    concepts = build_concepts(m)
    outcomes = pd.read_csv(TABLES / f"concept_outcomes_m{m}.csv")
    trends = pd.read_csv(TABLES / f"trends_m{m}.csv")
    slices = {
        "primary": build_slices(outcomes, trends, "primary"),
        "active": build_slices(outcomes, trends, "active"),
    }
    concept_records = concepts.astype(object).where(pd.notna(concepts), None).to_dict("records")
    return {
        "summary": {
            "m": m,
            "k": int(summary["k_active"]),
            "eligible": int(summary["n_eligible_startups"]),
            "features": int(summary["n_active_features"]),
            "medianPrimary": clean_number(summary["median_primary_area_size"], 1),
            "minimumPrimary": int(summary["minimum_primary_area_size"]),
            "silhouette": clean_number(summary["cosine_silhouette_primary_assignment"], 3),
            "reconstructionNrmse": clean_number(summary["reconstruction_nrmse"], 3),
            "smallOrUnstable": int(concepts["smallAndUnstable"].sum()),
            "seeds": [42, 7, 2024],
        },
        "concepts": concept_records,
        "slices": slices,
        "profiles": build_profiles(m, concepts, slices),
        "founderMatrix": build_founder_matrix(m),
        "disagreement": build_disagreement(m),
    }


def build_coverage() -> dict[str, Any]:
    coverage = pd.read_csv(TABLES / "year_track_coverage.csv")
    missing = pd.read_csv(TABLES / "problem_text_missingness.csv")
    missing["usable"] = missing["problem_text_quality"].isin(["high", "medium"])
    usable = (
        missing.groupby(["year", "Track", "usable"], as_index=False)["n_applications"].sum()
        .pivot_table(index=["year", "Track"], columns="usable", values="n_applications", fill_value=0)
        .reset_index()
    )
    records = []
    for _, row in coverage.iterrows():
        year = int(row["year"])
        track = str(row["Track"])
        matched = usable[(usable["year"] == year) & (usable["Track"] == track)]
        usable_n = int(matched[True].iloc[0]) if not matched.empty and True in matched else int(row["applications"])
        missing_n = int(row["applications"]) - usable_n
        records.append(
            {
                "year": year,
                "track": clean_label(track),
                "judgeRows": int(row["judge_rows"]),
                "applications": int(row["applications"]),
                "usableProblemText": usable_n,
                "missingProblemText": missing_n if missing_n >= MIN_CELL else None,
                "missingSuppressed": 0 < missing_n < MIN_CELL,
            }
        )
    return {"yearTrack": records}


def build() -> dict[str, Any]:
    comparison = pd.read_csv(TABLES / "concept_granularity_comparison.csv").set_index("m_concepts")
    audit = json.loads((TABLES / "audit_summary.json").read_text())
    models = {str(m): build_model(m, comparison.loc[m]) for m in (16, 32)}

    primary_all = models["16"]["slices"]["primary"]["All tracks"]["All years"]["items"]
    largest = max(primary_all, key=lambda item: item["n"])
    highest = max(primary_all, key=lambda item: item["recommendation"] or -999)
    fastest = max(primary_all, key=lambda item: item["trend"] or -999)
    widest_gap = max(primary_all, key=lambda item: (item["problem"] or 0) - (item["business"] or 0))

    payload: dict[str, Any] = {
        "meta": {
            "title": "Harvard i-lab Problem Landscape",
            "period": "2021–2024",
            "generatedFrom": "validated local research pipeline",
            "privacy": f"Aggregate cells only; cells below {MIN_CELL} observations are suppressed.",
            "minimumCell": MIN_CELL,
            "eligibleProblemTexts": int(comparison.loc[16, "n_eligible_startups"]),
            "applications": int(audit["applications"]),
            "judgeRows": int(audit["judge_rows"]),
            "includedJudgeRows": 7240,
            "missingProblemTexts": int(audit["applications"] - comparison.loc[16, "n_eligible_startups"]),
            "sourceGranularity": "judge-level",
            "defaultModel": 16,
            "defaultMembership": "primary",
            "embeddingModel": "nomic-ai/modernbert-embed-base",
            "upstreamCommit": "706d8979c71befe8e3151f51a8d4dbf87d4d7e41",
        },
        "models": models,
        "coverage": build_coverage(),
        "findings": [
            {
                "eyebrow": "Portfolio concentration",
                "value": f"{largest['n']} ventures",
                "title": largest["label"],
                "note": f"{largest['share'] * 100:.1f}% of ventures with usable problem text.",
            },
            {
                "eyebrow": "Highest mean recommendation",
                "value": f"{highest['recommendation']:.2f} / 5",
                "title": highest["label"],
                "note": f"Startup-weighted mean across {highest['n']} ventures; descriptive, not causal.",
            },
            {
                "eyebrow": "Fastest attention growth",
                "value": f"+{fastest['trend'] * 100:.1f} pts / year",
                "title": fastest["label"],
                "note": "Slope in yearly portfolio share across four annual observations.",
            },
            {
                "eyebrow": "Founder-support signal",
                "value": f"+{(widest_gap['problem'] - widest_gap['business']):.2f}",
                "title": widest_gap["label"],
                "note": "Average problem-definition score minus average business-model score.",
            },
        ],
    }

    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    forbidden = [
        '"application_id"',
        '"venture_name"',
        '"problem_text"',
        '"top_examples"',
        '"moderate_examples"',
        '"member_number"',
        '"reviewer_email"',
    ]
    assert not any(term in serialized.lower() for term in forbidden)
    for model in models.values():
        for membership in model["slices"].values():
            for scope in membership.values():
                for period in scope.values():
                    assert all(item["n"] >= MIN_CELL for item in period["items"])
        for matrix in model["founderMatrix"].values():
            assert all(cell["n"] >= MIN_CELL for cell in matrix)
    return payload


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote disclosure-controlled payload to {OUTPUT.relative_to(ROOT)}")
