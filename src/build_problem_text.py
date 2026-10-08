from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import PATHS


SOLUTION_LEAKAGE_TERMS = re.compile(
    r"\b(our solution|we (?:built|build|provide|offer|developed)|platform|app|software|saas|"
    r"ai-powered|machine[- ]learning|technology|product|device)\b",
    flags=re.IGNORECASE,
)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _word_count(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value))


def build_problem_record(row: pd.Series) -> dict[str, object]:
    hls_customer = _clean(row.get("stakeholders", ""))
    hls_problem = _clean(row.get("problem HLS", ""))
    general_customer = _clean(row.get("Customer", ""))
    general_problem = _clean(row.get("Problem", ""))

    hls_score = len(hls_customer) + len(hls_problem)
    general_score = len(general_customer) + len(general_problem)
    if hls_score == 0 and general_score == 0:
        source_form = "missing"
        customer = problem = ""
    elif hls_score >= general_score:
        source_form = "hls"
        customer, problem = hls_customer, hls_problem
    else:
        source_form = "open_social"
        customer, problem = general_customer, general_problem

    parts = []
    if customer:
        parts.append(f"Customer/Stakeholder: {customer}")
    if problem:
        parts.append(f"Problem/Need: {problem}")
    text = "\n".join(parts)
    words = _word_count(text)
    fields_present = int(bool(customer)) + int(bool(problem))
    if not text or words < 10:
        quality = "low"
    elif fields_present == 2 and words >= 30:
        quality = "high"
    else:
        quality = "medium"

    official_track = _clean(row.get("Track", ""))
    expected_form = "hls" if official_track == "Health & Life Science" else "open_social"
    return {
        "problem_text": text,
        "problem_text_source_form": source_form,
        "problem_text_quality": quality,
        "problem_text_word_count": words,
        "problem_text_missing": not bool(text),
        "track_text_source_mismatch": source_form not in ("missing", expected_form),
        "solution_leakage_flag": bool(SOLUTION_LEAKAGE_TERMS.search(text)),
    }


def build_problem_text_dataset(startup: pd.DataFrame) -> pd.DataFrame:
    records = startup.apply(build_problem_record, axis=1, result_type="expand")
    out = pd.concat([startup.copy(), records], axis=1)
    return out


def write_problem_text_outputs(problem_df: pd.DataFrame, output_dir: Path = PATHS.tables) -> None:
    PATHS.ensure()
    problem_df.to_csv(output_dir / "problem_text.csv", index=False)
    missing = (
        problem_df.groupby(["year", "Track", "problem_text_quality"], dropna=False)
        .size()
        .rename("n_applications")
        .reset_index()
    )
    totals = problem_df.groupby(["year", "Track"]).size().rename("applications_total").reset_index()
    missing = missing.merge(totals, on=["year", "Track"], how="left")
    missing["share"] = missing["n_applications"] / missing["applications_total"]
    missing.to_csv(output_dir / "problem_text_missingness.csv", index=False)
    exceptions = problem_df.loc[
        problem_df["track_text_source_mismatch"] | problem_df["problem_text_missing"] | problem_df["solution_leakage_flag"],
        [
            "application_id", "year", "Track", "venture_name", "problem_text_source_form",
            "problem_text_quality", "problem_text_missing", "track_text_source_mismatch", "solution_leakage_flag",
        ],
    ]
    exceptions.to_csv(output_dir / "problem_text_review_queue.csv", index=False)

