from __future__ import annotations

import json
from pathlib import Path

from src.public_privacy import assert_public_privacy


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "vercel-site"


def _payload() -> dict:
    return json.loads((SITE / "data" / "landscape.json").read_text())


def test_three_view_meeting_structure_and_atlas_link() -> None:
    html = (SITE / "index.html").read_text()
    assert html.count('class="view ') == 3
    assert 'id="shared"' in html
    assert 'id="explore"' in html
    assert 'id="support"' in html
    assert "01</span> Shared problems" in html
    assert "02</span> Explore a problem" in html
    assert "03</span> Where i-lab can help" in html
    assert 'href="/atlas"' in html
    assert (SITE / "atlas.html").exists()
    assert (SITE / "atlas.js").exists()
    assert (SITE / "atlas.css").exists()


def test_meeting_taxonomy_is_fixed_m16_and_default_is_education() -> None:
    payload = _payload()
    meeting = payload["meeting"]
    assert meeting["taxonomy"] == "M=16 primary assignment"
    assert len(meeting["problems"]) == 6
    default = next(item for item in meeting["problems"] if item["id"] == meeting["defaultProblemId"])
    assert default["label"] == "Equitable K–12 learning and student support"
    html = (SITE / "index.html").read_text()
    assert "model-control" not in html
    assert "membership-control" not in html


def test_suppressed_meeting_cells_never_look_like_zero() -> None:
    payload = _payload()
    suppressed = []
    for problem in payload["meeting"]["problems"]:
        suppressed.extend(cell for cell in problem["tracks"].values() if cell["suppressed"])
        suppressed.extend(cell for cell in problem["years"] if cell["suppressed"])
    assert suppressed
    assert all("n" not in cell and "share" not in cell for cell in suppressed)
    app = (SITE / "app.js").read_text()
    assert "N &lt; 10 · not zero" in app
    assert 'cell.suppressed' in app


def test_selected_problem_is_one_state_across_all_views() -> None:
    app = (SITE / "app.js").read_text()
    select_block = app[app.index("function selectProblem"):app.index("function heatStyle")]
    assert "state.selectedId = next.id" in select_block
    assert "renderSharedProblems()" in select_block
    assert "renderExplore()" in select_block
    assert "renderSupport()" in select_block
    assert 'searchParams.set("problem"' in select_block


def test_partial_2021_and_gender_evaluation_disclosure() -> None:
    payload = _payload()
    minimum = payload["meta"]["minimumCell"]
    for problem in payload["meeting"]["problems"]:
        year_2021 = next(item for item in problem["years"] if item["year"] == 2021)
        assert year_2021["coverage"] == "partial"
        assert year_2021["coverageLabel"] == "Partial problem-text coverage"
        evaluation = problem["leadGender"]["evaluation"]
        if evaluation["available"]:
            assert all(group["n"] >= minimum for group in evaluation["groups"])
        else:
            assert evaluation["suppressed"]
            assert "groups" not in evaluation
            assert "adjustedDifferenceFemaleMinusMale" not in evaluation


def test_support_is_framed_as_a_question_to_validate() -> None:
    payload = _payload()
    for problem in payload["meeting"]["problems"]:
        support = problem["support"]
        assert support["question"].endswith("?")
        assert "validate" in support["interpretation"].lower()
        assert "not a measured intervention effect" in support["interpretation"].lower()
        assert support["n"] >= payload["meta"]["minimumCell"]


def test_meeting_public_privacy_and_language_boundary() -> None:
    payload = _payload()
    assert_public_privacy(payload)
    public_text = "\n".join(
        [
            (SITE / "index.html").read_text(),
            (SITE / "app.js").read_text(),
            json.dumps(payload["meeting"]),
        ]
    ).casefold()
    for unsupported in (
        "women are penalized",
        "female penalty",
        "gender penalty",
        "causal bias",
        "market opportunity",
        "white space",
    ):
        assert unsupported not in public_text

