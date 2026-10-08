from __future__ import annotations

import pandas as pd

from src.build_problem_text import build_problem_record, build_problem_text_dataset


def test_open_problem_text_excludes_generic_description() -> None:
    row = pd.Series(
        {
            "Track": "Open",
            "Customer": "Hospital administrators",
            "Problem": "Nursing schedules are difficult to coordinate across shifts and preferences.",
            "problem HLS": "",
            "stakeholders": "",
            "Description": "AI-powered scheduling software",
        }
    )
    record = build_problem_record(row)
    assert record["problem_text_source_form"] == "open_social"
    assert "AI-powered" not in record["problem_text"]
    assert record["problem_text"].startswith("Customer/Stakeholder:")


def test_hls_routing_uses_populated_form_not_official_track() -> None:
    row = pd.Series(
        {
            "Track": "Open",
            "Customer": "",
            "Problem": "",
            "stakeholders": "Patients and clinicians",
            "problem HLS": "Existing therapies have severe side effects and low efficacy.",
        }
    )
    record = build_problem_record(row)
    assert record["problem_text_source_form"] == "hls"
    assert record["track_text_source_mismatch"]


def test_missing_structured_fields_are_flagged_not_backfilled() -> None:
    frame = pd.DataFrame(
        [{"Track": "Social Impact", "Customer": "", "Problem": "", "stakeholders": "", "problem HLS": "", "Description": "A product description"}]
    )
    result = build_problem_text_dataset(frame)
    assert result.loc[0, "problem_text"] == ""
    assert bool(result.loc[0, "problem_text_missing"])
    assert result.loc[0, "problem_text_quality"] == "low"
