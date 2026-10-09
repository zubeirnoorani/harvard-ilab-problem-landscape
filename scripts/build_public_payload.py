"""Build the disclosure-controlled payload used by the hosted research atlas.

Only aggregate cells with at least ``MIN_CELL`` observations are published.
Application text, venture names, identifiers, row-level demographics, embeddings,
activations, checkpoints, and example excerpts never enter the hosted payload.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import concept_quality_badge  # noqa: E402
from src.public_privacy import assert_public_privacy  # noqa: E402

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

MEETING_TRACKS = ("Open", "Social Impact", "Health & Life Sciences")
DEFAULT_MEETING_PROBLEM = "Equitable K–12 learning and student support"
SUPPORT_HYPOTHESES: dict[int, dict[str, str]] = {
    5: {
        "theme": "Employer discovery / workforce pilots",
        "question": "Could shared employer and workforce-partner sessions help teams test who owns the budget, what adoption requires, and what makes a pilot credible?",
        "validation": "Ask founders whether employer access, budget ownership, or evidence for a pilot is actually the binding constraint.",
    },
    7: {
        "theme": "Buyer discovery / pilot partnerships",
        "question": "Could shared buyer-discovery and pilot-partner access help education teams test who buys, how procurement works, and what a useful pilot looks like?",
        "validation": "Ask founders whether buyer access and procurement are actually the binding constraints.",
    },
    8: {
        "theme": "Clinical pathways / hospital partners",
        "question": "Could shared clinical and hospital-partner sessions help teams test an adoption pathway?",
        "validation": "Ask teams and clinical partners whether workflow fit, evidence, procurement, or another constraint is most important.",
    },
    11: {
        "theme": "Deployment partners / operating economics",
        "question": "Could shared introductions to deployment partners help teams test adoption and operating economics?",
        "validation": "Ask founders whether partner access, unit economics, project finance, or regulation is the main constraint.",
    },
    14: {
        "theme": "Workflow discovery / design partners",
        "question": "Could structured design-partner sessions help teams test how a proposed change fits existing workflows, budget ownership, and adoption?",
        "validation": "Ask founders and operators whether workflow access, integration, change management, or buyer clarity is the binding constraint.",
    },
    15: {
        "theme": "Care pathways / payer and caregiver discovery",
        "question": "Could shared access to care-delivery partners help teams test where they fit in patient and caregiver journeys, and who pays?",
        "validation": "Ask founders whether pathway fit, reimbursement, clinical evidence, caregiver adoption, or another constraint is most important.",
    },
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
        "recommendationApplications": clean_number(row["recommendation_n_startups"]),
        "recommendationCiLow": clean_number(row["recommendation_ci_low"]),
        "recommendationCiHigh": clean_number(row["recommendation_ci_high"]),
        "adjustedRecommendation": clean_number(row["recommendation_adjusted_year_track_mean"]),
        "adjustedRecommendationSd": clean_number(row["recommendation_adjusted_year_track_sd_between_applications"]),
        "adjustedRecommendationApplications": clean_number(row["recommendation_adjusted_year_track_n_applications"]),
        "adjustedRecommendationCiLow": clean_number(row["recommendation_adjusted_year_track_ci_low"]),
        "adjustedRecommendationCiHigh": clean_number(row["recommendation_adjusted_year_track_ci_high"]),
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
        "trend2022To2024": None,
        "change2022To2024": None,
        "trend2021To2024Partial": None,
        "change2021To2024Partial": None,
        "trendHeadlineEligible": False,
    }
    if trend is not None:
        record["trend"] = clean_number(trend["trend_slope_2022_2024"])
        record["change"] = clean_number(trend["change_2022_to_2024"])
        record["trend2022To2024"] = clean_number(trend["trend_slope_2022_2024"])
        record["change2022To2024"] = clean_number(trend["change_2022_to_2024"])
        record["trend2021To2024Partial"] = clean_number(trend["trend_slope_2021_2024_partial"])
        record["change2021To2024Partial"] = clean_number(trend["change_2021_to_2024_partial"])
        record["trendHeadlineEligible"] = bool(
            int(trend["count_2022"]) >= MIN_CELL and int(trend["count_2024"]) >= MIN_CELL
        )
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
        "leadGender": (pd.read_csv(TABLES / f"lead_gender_composition_m{m}.csv"), "lead_gender", "n_applications"),
        "teamGender": (pd.read_csv(TABLES / f"team_gender_composition_m{m}.csv"), "gender", "n_founders"),
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
    track_index = pd.read_csv(TABLES / f"track_overrepresentation_m{m}.csv")
    school_index = pd.read_csv(TABLES / f"school_representation_m{m}.csv")
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
                "applications": int(row["n_startups"]),
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
                "problemMinusPrototype": clean_number(row["problem_minus_prototype"]),
                "prototypeMinusProblem": clean_number(row["prototype_minus_problem"]),
                "impactMinusBusiness": clean_number(row["impact_minus_business"]),
                "problem": clean_number(row["problem_customer_definition_mean_mean"]),
                "solution": clean_number(row["solution_prototype_mean_mean"]),
                "business": clean_number(row["business_model_mean_mean"]),
                "impact": clean_number(row["impact_mean_mean"]),
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
                        "coverage": "partial" if year == 2021 else "complete",
                        "coverageLabel": "Partial problem-text coverage" if year == 2021 else "Effectively complete problem-text coverage",
                    }
                )
            trend_record = {
                "slope": clean_number(row["trend_slope_2022_2024"]),
                "change": clean_number(row["change_2022_to_2024"]),
                "slope2022To2024": clean_number(row["trend_slope_2022_2024"]),
                "change2022To2024": clean_number(row["change_2022_to_2024"]),
                "slope2021To2024Partial": clean_number(row["trend_slope_2021_2024_partial"]),
                "change2021To2024Partial": clean_number(row["change_2021_to_2024_partial"]),
                "years": year_values,
            }

        track_records = [
            {
                "track": clean_label(row["Track"]),
                "n": int(row["n_applications"]),
                "trackShare": clean_number(row["track_share"]),
                "overallShare": clean_number(row["overall_share"]),
                "ratio": clean_number(row["overrepresentation_ratio"]),
                "shareDifference": clean_number(row["share_difference"]),
            }
            for _, row in track_index.loc[
                track_index["concept_id"].eq(cid) & track_index["n_applications"].ge(MIN_CELL)
            ].sort_values("overrepresentation_ratio", ascending=False).iterrows()
        ]
        school_records = [
            {
                "school": clean_label(row["harvard_school"]),
                "n": int(row["n_founders"]),
                "representationIndex": clean_number(row["representation_index"]),
                "problemShare": clean_number(row["share_of_problem_from_school"]),
                "portfolioShare": clean_number(row["share_of_all_reported_founders_from_school"]),
            }
            for _, row in school_index.loc[
                school_index["concept_id"].eq(cid) & school_index["n_founders"].ge(MIN_CELL)
            ].sort_values("representation_index", ascending=False).iterrows()
        ]

        profiles[str(cid)] = {
            "primaryOutcome": primary,
            "activeOutcome": active_all.get(cid),
            "support": support,
            "trend": trend_record,
            "trackOverrepresentation": track_records,
            "schoolRepresentation": school_records,
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
        quality_badge = concept_quality_badge(stable, row)
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
                "qualityBadge": quality_badge,
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
        "teamGender": ("team_gender_composition", "gender", "n_founders"),
        "leadGender": ("lead_gender_composition", "lead_gender", "n_applications"),
        "schoolCounts": ("composition_harvard_school", "harvard_school", "n_founders"),
        "countries": ("composition_country", "country"),
    }
    for name, spec in specs.items():
        file_stem, category = spec[:2]
        count_col = spec[2] if len(spec) == 3 else "n_founders"
        data = pd.read_csv(TABLES / f"{file_stem}_m{m}.csv")
        safe = data[data[count_col] >= MIN_CELL].copy()
        result[name] = [
            {
                "conceptId": int(row["concept_id"]),
                "conceptLabel": str(row["concept_label"]),
                "category": clean_label(row[category]),
                "n": int(row[count_col]),
                **(
                    {
                        "reportingCoverage": clean_number(row["reporting_coverage"]),
                        "shareAmongReported": clean_number(row["share_among_reported"]),
                    }
                    if "reporting_coverage" in row.index
                    else {}
                ),
            }
            for _, row in safe.iterrows()
        ]
    school_index = pd.read_csv(TABLES / f"school_representation_m{m}.csv")
    school_index = school_index[school_index["n_founders"] >= MIN_CELL]
    result["schoolRepresentation"] = [
        {
            "conceptId": int(row["concept_id"]),
            "conceptLabel": str(row["concept_label"]),
            "category": clean_label(row["harvard_school"]),
            "n": int(row["n_founders"]),
            "representationIndex": clean_number(row["representation_index"]),
            "problemShare": clean_number(row["share_of_problem_from_school"]),
            "portfolioShare": clean_number(row["share_of_all_reported_founders_from_school"]),
        }
        for _, row in school_index.iterrows()
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
    lead = pd.read_csv(TABLES / f"lead_gender_composition_m{m}.csv")
    team = pd.read_csv(TABLES / f"team_gender_composition_m{m}.csv")
    lead_totals = lead.drop_duplicates("concept_id")
    team_totals = team.drop_duplicates("concept_id")
    result["genderCoverage"] = {
        "lead": {
            "reported": int(lead_totals["reported_lead_gender_n"].sum()),
            "total": int(lead_totals["problem_n_applications"].sum()),
            "coverage": clean_number(
                lead_totals["reported_lead_gender_n"].sum() / lead_totals["problem_n_applications"].sum()
            ),
        },
        "team": {
            "reported": int(team_totals["team_gender_reported_n"].sum()),
            "total": int(team_totals["team_records_total"].sum()),
            "coverage": clean_number(
                team_totals["team_gender_reported_n"].sum() / team_totals["team_records_total"].sum()
            ),
        },
    }
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
                "meanWithinApplicationSd": clean_number(row["mean_within_startup_recommendation_sd"]),
                "medianWithinApplicationSd": clean_number(row["median_within_startup_recommendation_sd"]),
                "n": int(row["n_startups"]),
                "ratings": int(row["n_underlying_ratings"]),
                "quadrant": str(row["rating_disagreement_quadrant"]),
            }
            for _, row in rows.iterrows()
        ]
    return result


def build_track_signatures(m: int, concepts: pd.DataFrame) -> dict[str, Any]:
    data = pd.read_csv(TABLES / f"track_overrepresentation_m{m}.csv")
    quality = concepts.set_index("id")
    result: dict[str, Any] = {}
    for display, (_, raw_track, _) in SCOPE_MAP.items():
        if raw_track is None:
            continue
        rows = data.loc[data["Track"].eq(raw_track) & data["n_applications"].ge(MIN_CELL)].copy()
        rows = rows.loc[
            rows["concept_id"].map(lambda cid: quality.loc[int(cid), "qualityBadge"] in {"Stable", "Stable but broad"})
        ]
        result[display] = [
            {
                "id": int(row["concept_id"]),
                "label": str(row["concept_label"]),
                "n": int(row["n_applications"]),
                "trackShare": clean_number(row["track_share"]),
                "overallShare": clean_number(row["overall_share"]),
                "ratio": clean_number(row["overrepresentation_ratio"]),
                "shareDifference": clean_number(row["share_difference"]),
            }
            for _, row in rows.sort_values(
                ["overrepresentation_ratio", "n_applications"], ascending=False
            ).head(4).iterrows()
        ]
    return result


def build_support_portfolio(m: int) -> list[dict[str, Any]]:
    data = pd.read_csv(TABLES / f"intervention_profiles_m{m}.csv")
    safe = data.loc[data["n_startups"].ge(MIN_CELL)].copy()
    return [
        {
            "id": int(row["concept_id"]),
            "label": str(row["concept_label"]),
            "n": int(row["n_startups"]),
            "problem": clean_number(row["problem_customer_definition_mean_mean"]),
            "solution": clean_number(row["solution_prototype_mean_mean"]),
            "business": clean_number(row["business_model_mean_mean"]),
            "impact": clean_number(row["impact_mean_mean"]),
            "problemMinusBusiness": clean_number(row["problem_minus_business"]),
            "problemMinusPrototype": clean_number(row["problem_minus_prototype"]),
            "impactMinusBusiness": clean_number(row["impact_minus_business"]),
            "pattern": str(row["support_pattern"]),
        }
        for _, row in safe.sort_values("problem_minus_business", ascending=False).iterrows()
    ]


def build_meeting_view(leadership: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed M=16, primary-assignment narrative used in the meeting."""
    primary = leadership["slices"]["primary"]["All tracks"]["All years"]["items"]
    shared = sorted(primary, key=lambda item: item["n"], reverse=True)[:6]
    shared_ids = {int(item["id"]) for item in shared}
    concepts = {int(item["id"]): item for item in leadership["concepts"]}
    tracks = pd.read_csv(TABLES / "track_overrepresentation_m16.csv")
    trends = pd.read_csv(TABLES / "trends_m16.csv")
    schools = pd.read_csv(TABLES / "composition_harvard_school_m16.csv")
    lead_gender = pd.read_csv(TABLES / "lead_gender_composition_m16.csv")
    gender_evaluation = pd.read_csv(TABLES / "gender_problem_evaluation_m16.csv")

    track_name = {
        "Open": "Open",
        "Social Impact": "Social Impact",
        "Health & Life Science": "Health & Life Sciences",
    }
    track_denominators = {
        track_name[str(row["Track"])]: int(row["track_n_applications"])
        for _, row in tracks.drop_duplicates("Track").iterrows()
    }
    records: list[dict[str, Any]] = []
    for item in shared:
        cid = int(item["id"])
        meta = concepts[cid]
        profile = leadership["profiles"][str(cid)]

        track_cells: dict[str, dict[str, Any]] = {}
        for raw, display in track_name.items():
            row = tracks.loc[tracks["concept_id"].eq(cid) & tracks["Track"].eq(raw)].iloc[0]
            n = int(row["n_applications"])
            track_cells[display] = (
                {
                    "suppressed": False,
                    "n": n,
                    "share": clean_number(row["track_share"]),
                }
                if n >= MIN_CELL
                else {"suppressed": True}
            )

        trend = trends.loc[trends["track_scope"].eq("All tracks") & trends["concept_id"].eq(cid)].iloc[0]
        year_cells: list[dict[str, Any]] = []
        for year in YEARS:
            n = int(trend[f"count_{year}"])
            year_cells.append(
                {
                    "year": year,
                    "coverage": "partial" if year == 2021 else "complete",
                    "coverageLabel": "Partial problem-text coverage" if year == 2021 else "Complete/near-complete problem-text coverage",
                    **(
                        {"suppressed": False, "n": n, "share": clean_number(trend[f"share_{year}"])}
                        if n >= MIN_CELL
                        else {"suppressed": True}
                    ),
                }
            )

        school_rows = schools.loc[
            schools["concept_id"].eq(cid) & schools["n_founders"].ge(MIN_CELL)
        ].sort_values("n_founders", ascending=False)
        school_cells = [
            {
                "label": clean_label(row["harvard_school"]),
                "n": int(row["n_founders"]),
                "share": clean_number(row["share_within_problem"]),
                "unit": "founder records",
            }
            for _, row in school_rows.iterrows()
        ]

        lead_rows = lead_gender.loc[lead_gender["concept_id"].eq(cid)].copy()
        reported_n = int(lead_rows["reported_lead_gender_n"].iloc[0])
        problem_n = int(lead_rows["problem_n_applications"].iloc[0])
        gender_cells: list[dict[str, Any]] = []
        for label in ("Female", "Male"):
            row = lead_rows.loc[lead_rows["lead_gender"].astype(str).str.casefold().eq(label.casefold())]
            if not row.empty and int(row.iloc[0]["n_applications"]) >= MIN_CELL:
                gender_cells.append(
                    {
                        "label": label,
                        "n": int(row.iloc[0]["n_applications"]),
                        "shareAmongReported": clean_number(row.iloc[0]["share_among_reported"]),
                    }
                )

        evaluation_row = gender_evaluation.loc[gender_evaluation["concept_id"].eq(cid)]
        evaluation: dict[str, Any]
        if not evaluation_row.empty and bool(evaluation_row.iloc[0]["comparison_publishable"]):
            row = evaluation_row.iloc[0]
            evaluation = {
                "available": True,
                "differenceLabel": "Evaluation difference within this problem",
                "adjustedDifferenceFemaleMinusMale": clean_number(row["adjusted_difference_female_minus_male"]),
                "adjustedDifferenceCiLow": clean_number(row["adjusted_difference_ci_low"]),
                "adjustedDifferenceCiHigh": clean_number(row["adjusted_difference_ci_high"]),
                "groups": [
                    {
                        "label": "Female",
                        "n": int(row["female_n"]),
                        "rawMean": clean_number(row["female_raw_mean"]),
                        "rawCiLow": clean_number(row["female_raw_ci_low"]),
                        "rawCiHigh": clean_number(row["female_raw_ci_high"]),
                        "adjustedMean": clean_number(row["female_adjusted_mean"]),
                        "adjustedCiLow": clean_number(row["female_adjusted_ci_low"]),
                        "adjustedCiHigh": clean_number(row["female_adjusted_ci_high"]),
                    },
                    {
                        "label": "Male",
                        "n": int(row["male_n"]),
                        "rawMean": clean_number(row["male_raw_mean"]),
                        "rawCiLow": clean_number(row["male_raw_ci_low"]),
                        "rawCiHigh": clean_number(row["male_raw_ci_high"]),
                        "adjustedMean": clean_number(row["male_adjusted_mean"]),
                        "adjustedCiLow": clean_number(row["male_adjusted_ci_low"]),
                        "adjustedCiHigh": clean_number(row["male_adjusted_ci_high"]),
                    },
                ],
            }
        else:
            evaluation = {
                "available": False,
                "suppressed": True,
                "message": "Comparison suppressed because at least one lead-gender cell has fewer than 10 applications.",
            }

        support = profile["support"]
        hypothesis = SUPPORT_HYPOTHESES[cid]
        records.append(
            {
                "id": cid,
                "label": str(item["label"]),
                "description": str(meta["description"]),
                "n": int(item["n"]),
                "share": clean_number(item["share"]),
                "tracks": track_cells,
                "years": year_cells,
                "schools": school_cells,
                "schoolSuppressedCells": int((schools.loc[schools["concept_id"].eq(cid), "n_founders"] < MIN_CELL).sum()),
                "leadGender": {
                    "reportedN": reported_n,
                    "problemN": problem_n,
                    "reportingCoverage": clean_number(reported_n / problem_n if problem_n else np.nan),
                    "groups": gender_cells,
                    "evaluation": evaluation,
                },
                "support": {
                    "n": int(support["n"]),
                    "problem": clean_number(support["problem"]),
                    "business": clean_number(support["business"]),
                    "gap": clean_number(support["problemMinusBusiness"]),
                    "observed": "Problem & Customer Definition is higher than Business Model in historical application ratings.",
                    "theme": hypothesis["theme"],
                    "question": hypothesis["question"],
                    "validation": hypothesis["validation"],
                    "interpretation": "Proposed interpretation to validate; not a measured intervention effect.",
                },
            }
        )

    default_id = next(
        (record["id"] for record in records if record["label"] == DEFAULT_MEETING_PROBLEM),
        records[0]["id"],
    )
    return {
        "taxonomy": "M=16 primary assignment",
        "defaultProblemId": default_id,
        "fullAtlasRoute": "/atlas",
        "trackDenominators": track_denominators,
        "problems": records,
        "narrative": "The same underlying customer problems recur across PIC tracks, cohorts, Harvard communities, and applicant groups.",
    }


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
        "trackSignatures": build_track_signatures(m, concepts),
        "supportPortfolio": build_support_portfolio(m),
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
                "coverageStatus": "partial" if year == 2021 else "effectively_complete",
                "coverageLabel": "Partial problem-text coverage" if year == 2021 else "Effectively complete problem-text coverage",
            }
        )
    return {
        "yearTrack": records,
        "yearLabels": {
            "2021": "Partial problem-text coverage",
            "2022": "Effectively complete problem-text coverage",
            "2023": "Effectively complete problem-text coverage",
            "2024": "Effectively complete problem-text coverage",
        },
    }


def build() -> dict[str, Any]:
    comparison = pd.read_csv(TABLES / "concept_granularity_comparison.csv").set_index("m_concepts")
    audit = json.loads((TABLES / "audit_summary.json").read_text())
    models = {str(m): build_model(m, comparison.loc[m]) for m in (16, 32)}

    leadership = models["16"]
    concept_lookup = {item["id"]: item for item in leadership["concepts"]}
    primary_all = leadership["slices"]["primary"]["All tracks"]["All years"]["items"]
    headline_pool = [
        item
        for item in primary_all
        if concept_lookup[item["id"]]["qualityBadge"] in {"Stable", "Stable but broad"}
        and not concept_lookup[item["id"]]["smallAndUnstable"]
    ]
    largest = max(headline_pool, key=lambda item: item["n"])
    fastest = max(
        [item for item in headline_pool if item["trend2022To2024"] is not None and item["trendHeadlineEligible"]],
        key=lambda item: item["trend2022To2024"],
    )
    adjusted_median = float(np.median([item["adjustedRecommendation"] for item in headline_pool]))
    prevalence_median = float(np.median([item["share"] for item in headline_pool]))
    stronger_lower_attention = sorted(
        [
            item
            for item in headline_pool
            if item["adjustedRecommendation"] > adjusted_median and item["share"] < prevalence_median
        ],
        key=lambda item: item["adjustedRecommendation"],
        reverse=True,
    )
    support_pool = [
        item for item in leadership["supportPortfolio"] if item["id"] in {row["id"] for row in headline_pool}
    ]
    support_ranked = sorted(
        support_pool,
        key=lambda item: max(item["problemMinusBusiness"], item["impactMinusBusiness"]),
        reverse=True,
    )
    disagreement_ranked = sorted(
        [item for item in leadership["disagreement"]["All tracks"] if item["id"] in {row["id"] for row in headline_pool}],
        key=lambda item: item["meanWithinApplicationSd"],
        reverse=True,
    )
    stronger = stronger_lower_attention[0]
    support = support_ranked[0]

    payload: dict[str, Any] = {
        "meta": {
            "title": "Harvard i-lab Problem Landscape",
            "period": "2021–2024",
            "defaultTrendPeriod": "2022–2024",
            "partialCoveragePeriod": "2021–2024 (2021 partial problem-text coverage)",
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
            "defaultSchoolView": "representation_index",
            "defaultGenderView": "lead_applicant",
            "embeddingModel": "nomic-ai/modernbert-embed-base",
            "upstreamCommit": "706d8979c71befe8e3151f51a8d4dbf87d4d7e41",
            "bootstrap": {"method": "application-level percentile bootstrap", "iterations": 2000, "seed": 42, "confidence": 0.95},
        },
        "models": models,
        "meeting": build_meeting_view(leadership),
        "coverage": build_coverage(),
        "findings": [
            {
                "signal": "Largest established problem area",
                "id": largest["id"],
                "title": largest["label"],
                "value": f"{largest['n']} applications",
                "n": largest["n"],
                "evidence": f"{largest['share'] * 100:.1f}% of applications with usable problem text use this as their primary area.",
                "implication": "Use this as a baseline for where founder attention and i-lab exposure are already concentrated.",
            },
            {
                "signal": "Fastest established 2022–2024 attention increase",
                "id": fastest["id"],
                "title": fastest["label"],
                "value": f"{fastest['change2022To2024'] * 100:+.1f} pts",
                "n": fastest["n"],
                "evidence": f"Largest portfolio-share increase among areas with at least 10 applications in both endpoint years; N={fastest['n']} applications across all years.",
                "implication": "Use the shift to prompt qualitative follow-up on changing founder needs, not as a forecast.",
            },
            {
                "signal": "Stronger evaluation, lower historical attention",
                "id": stronger["id"],
                "title": stronger["label"],
                "value": f"{stronger['adjustedRecommendation']:+.2f} adjusted",
                "n": stronger["n"],
                "evidence": f"Above-median year×track-adjusted Recommendation and below-median prevalence; raw {stronger['recommendation']:.2f}/5, N={stronger['n']} applications.",
                "implication": "Treat this as a candidate for qualitative follow-up, not a market or investment claim.",
            },
            {
                "signal": "Largest problem-vs-business-model gap",
                "id": support["id"],
                "title": support["label"],
                "value": f"{support['problemMinusBusiness']:+.2f}",
                "n": support["n"],
                "evidence": f"Problem & Customer Definition minus Business Model across N={support['n']} applications.",
                "implication": "Use the historical rubric profile to frame mentoring or curriculum questions; it does not identify a causal remedy.",
            },
        ],
        "actionableSignals": {
            "strongerEvaluationLowerAttention": [
                {
                    "id": item["id"], "label": item["label"], "n": item["n"],
                    "share": item["share"], "rawRecommendation": item["recommendation"],
                    "adjustedRecommendation": item["adjustedRecommendation"],
                    "adjustedCiLow": item["adjustedRecommendationCiLow"],
                    "adjustedCiHigh": item["adjustedRecommendationCiHigh"],
                }
                for item in stronger_lower_attention[:4]
            ],
            "founderSupportGaps": support_ranked[:4],
            "judgeDisagreement": disagreement_ranked[:4],
        },
    }

    assert_public_privacy(payload)
    for model in models.values():
        for membership in model["slices"].values():
            for scope in membership.values():
                for period in scope.values():
                    assert all(item["n"] >= MIN_CELL for item in period["items"])
        for matrix in model["founderMatrix"].values():
            if isinstance(matrix, list):
                assert all(cell["n"] >= MIN_CELL for cell in matrix)
        for signature in model["trackSignatures"].values():
            assert all(cell["n"] >= MIN_CELL for cell in signature)
        assert all(cell["n"] >= MIN_CELL for cell in model["supportPortfolio"])
    return payload


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote disclosure-controlled payload to {OUTPUT.relative_to(ROOT)}")
