from __future__ import annotations

import pytest

from src.aggregate_startups import build_startup_level
from src.build_problem_text import build_problem_text_dataset
from src.config import PATHS
from src.load_data import load_raw_data, validate_core_shape


@pytest.mark.skipif(not PATHS.raw_data.exists(), reason="Confidential source is not present")
def test_current_export_invariants() -> None:
    raw = load_raw_data()
    validate_core_shape(raw)
    startup, judge, issues = build_startup_level(raw)
    problem = build_problem_text_dataset(startup)
    assert len(raw) == 7281
    assert len(startup) == 509
    assert startup["application_id"].is_unique
    assert issues.empty
    assert int(problem["problem_text_missing"].sum()) == 50
    assert int((~problem["problem_text_missing"]).sum()) == 459
    retained = judge.loc[judge["included_in_aggregation"]]
    assert retained.groupby(["application_id", "judge_id"]).size().max() == 1
    assert int(judge["duplicate_application_judge"].sum()) == 4
