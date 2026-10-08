from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aggregate_startups import build_startup_level
from src.analysis import (
    _summarize_outcomes,
    build_track_overrepresentation,
    build_trends,
    lead_gender_composition,
    team_gender_composition,
)
from src.public_privacy import FORBIDDEN_PUBLIC_KEYS, assert_public_privacy


ROOT = Path(__file__).resolve().parents[1]


def _primary_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, count_a in zip((2021, 2022, 2023, 2024), (1, 2, 3, 4)):
        for index in range(10):
            is_a = index < count_a
            rows.append(
                {
                    "application_id": f"{year}-{index}",
                    "year": year,
                    "Track": "Open",
                    "concept_id": 0 if is_a else 1,
                    "concept_label": "Area A" if is_a else "Area B",
                }
            )
    return pd.DataFrame(rows)


def test_trends_default_to_2022_2024_and_label_2021_partial() -> None:
    trends = build_trends(_primary_fixture(), 16)
    row = trends.loc[(trends["track_scope"] == "All tracks") & (trends["concept_id"] == 0)].iloc[0]
    assert row["trend_slope_2022_2024"] == pytest.approx(0.1)
    assert row["change_2022_to_2024"] == pytest.approx(0.2)
    assert row["trend_slope_2021_2024_partial"] == pytest.approx(0.1)
    assert row["change_2021_to_2024_partial"] == pytest.approx(0.3)
    assert row["coverage_2021"] == "Partial problem-text coverage"
    assert row["default_trend_period"] == "2022–2024"


def _judge_row(application: str, reviewer: str, rating: int) -> dict[str, str]:
    return {
        "Submission ID": application,
        "year": "2024",
        "Track": "Open",
        "Account Name": application,
        "Reviewer Email Address": reviewer,
        "Contact ID - 18 Digit": "",
        "I have a conflict of interest:": "0",
        "Recommendation": str(rating),
        "Problem &amp; Customer Definition": str(rating),
        "Prototype/MVP": str(rating),
        "Business Model": str(rating),
        "Impact": str(rating),
    }


def test_year_track_adjustment_and_application_weighting() -> None:
    judge_rows = pd.DataFrame(
        [
            _judge_row("A", "a1@example.org", 5),
            _judge_row("A", "a2@example.org", 5),
            _judge_row("A", "a3@example.org", 5),
            _judge_row("B", "b1@example.org", 1),
        ]
    )
    applications, _, _ = build_startup_level(judge_rows)
    indexed = applications.set_index("application_id")
    assert indexed.loc["A", "recommendation_adjusted_year_track"] == pytest.approx(2.0)
    assert indexed.loc["B", "recommendation_adjusted_year_track"] == pytest.approx(-2.0)

    frame = applications.assign(concept_id=0, concept_label="Area")
    summary = _summarize_outcomes(frame, ["concept_id", "concept_label"]).iloc[0]
    assert summary["recommendation_mean"] == pytest.approx(3.0)
    assert summary["recommendation_mean"] != pytest.approx(4.0)  # Judge-row weighted mean.
    assert summary["recommendation_n_ratings"] == 4
    assert summary["n_applications"] == 2


def test_track_overrepresentation_uses_track_and_portfolio_denominators() -> None:
    rows = []
    for track, counts in {"Open": (6, 4), "Social Impact": (2, 8)}.items():
        for concept_id, count in enumerate(counts):
            for index in range(count):
                rows.append(
                    {
                        "application_id": f"{track}-{concept_id}-{index}",
                        "Track": track,
                        "concept_id": concept_id,
                        "concept_label": f"Area {concept_id}",
                    }
                )
    result = build_track_overrepresentation(pd.DataFrame(rows), 16)
    row = result.loc[(result["Track"] == "Open") & (result["concept_id"] == 0)].iloc[0]
    assert row["track_share"] == pytest.approx(0.6)
    assert row["overall_share"] == pytest.approx(0.4)
    assert row["overrepresentation_ratio"] == pytest.approx(1.5)
    assert row["share_difference"] == pytest.approx(0.2)


def test_lead_and_team_gender_are_distinct_with_coverage() -> None:
    primary = pd.DataFrame(
        [
            {"application_id": "A", "concept_id": 0, "concept_label": "Area", "Gender": "Female"},
            {"application_id": "B", "concept_id": 0, "concept_label": "Area", "Gender": ""},
        ]
    )
    founders = pd.DataFrame(
        [
            {"concept_id": 0, "concept_label": "Area", "gender": "Female"},
            {"concept_id": 0, "concept_label": "Area", "gender": "Male"},
            {"concept_id": 0, "concept_label": "Area", "gender": "Missing"},
        ]
    )
    lead = lead_gender_composition(primary, 16)
    team = team_gender_composition(founders, 16)
    assert lead["problem_n_applications"].iloc[0] == 2
    assert lead["reported_lead_gender_n"].iloc[0] == 1
    assert lead["reporting_coverage"].iloc[0] == pytest.approx(0.5)
    assert team["team_records_total"].iloc[0] == 3
    assert team["team_gender_reported_n"].iloc[0] == 2
    assert team["reporting_coverage"].iloc[0] == pytest.approx(2 / 3)


def test_committed_public_payload_privacy_defaults_and_headlines() -> None:
    payload = json.loads((ROOT / "vercel-site" / "data" / "landscape.json").read_text())
    assert_public_privacy(payload)
    assert payload["meta"]["defaultModel"] == 16
    assert payload["meta"]["defaultTrendPeriod"] == "2022–2024"
    assert payload["coverage"]["yearLabels"]["2021"] == "Partial problem-text coverage"

    concepts = {item["id"]: item for item in payload["models"]["16"]["concepts"]}
    for finding in payload["findings"]:
        combined = " ".join(str(finding.get(key, "")) for key in ("value", "evidence"))
        assert "application" in combined.lower()
        assert not re.search(r"\b\d+[\d,]*\s+ventures?\b", combined, flags=re.IGNORECASE)
        assert not concepts[finding["id"]]["smallAndUnstable"]
        assert concepts[finding["id"]]["qualityBadge"] in {"Stable", "Stable but broad"}

    minimum = payload["meta"]["minimumCell"]
    for model in payload["models"].values():
        for value in model["founderMatrix"].values():
            if isinstance(value, list):
                assert all(cell["n"] >= minimum for cell in value)


def test_privacy_guard_rejects_confidential_fields() -> None:
    for key in sorted(FORBIDDEN_PUBLIC_KEYS):
        with pytest.raises(AssertionError):
            assert_public_privacy({key: "confidential"})
