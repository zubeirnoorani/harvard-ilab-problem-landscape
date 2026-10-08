from __future__ import annotations

import pandas as pd

from src.aggregate_startups import build_startup_level, coerce_rating
from src.config import RATING_COLUMNS


def _row(application: str, reviewer: str, recommendation: str, coi: str = "0") -> dict[str, str]:
    row = {
        "Submission ID": application,
        "year": "2024",
        "Track": "Open",
        "Account Name": f"Venture {application}",
        "Venture Name": "",
        "Reviewer Email Address": reviewer,
        "Contact ID - 18 Digit": "",
        "reviewer_name": reviewer,
        "judge_type": "Founder",
        "Judge Type": "Founder",
        "I have a conflict of interest:": coi,
        "Recommendation": recommendation,
        "Problem &amp; Customer Definition": "4",
        "Prototype/MVP": "3",
        "Business Model": "2",
        "Impact": "5",
        "Problem": "A clear customer problem.",
        "Customer": "A defined customer.",
        "problem HLS": "",
        "stakeholders": "",
        "School": "HBS",
        "Gender": "Woman",
    }
    return row


def test_rating_coercion_excludes_zero_and_out_of_range() -> None:
    values = coerce_rating(pd.Series(["", "0", "1", "5", "6", "bad"]))
    assert values.notna().sum() == 2
    assert values.dropna().tolist() == [1.0, 5.0]


def test_startup_aggregation_excludes_conflicts_and_preserves_metadata() -> None:
    frame = pd.DataFrame(
        [
            _row("A", "one@example.org", "5"),
            _row("A", "two@example.org", "3"),
            _row("A", "conflict@example.org", "1", coi="1"),
            _row("B", "one@example.org", "0"),
        ]
    )
    startup, judge, issues = build_startup_level(frame)
    assert len(startup) == 2
    assert issues.empty
    first = startup.set_index("application_id").loc["A"]
    assert first["recommendation_mean"] == 4.0
    assert first["recommendation_n_judges"] == 2
    assert first["School"] == "HBS"
    assert len(judge) == 4
    assert judge["conflict_of_interest"].sum() == 1


def test_every_rating_dimension_has_four_aggregates() -> None:
    frame = pd.DataFrame([_row("A", "one@example.org", "4")])
    startup, _, _ = build_startup_level(frame)
    for dimension in RATING_COLUMNS:
        for suffix in ("mean", "sd", "median", "n_judges"):
            assert f"{dimension}_{suffix}" in startup.columns
