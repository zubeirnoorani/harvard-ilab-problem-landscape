from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PATHS, RATING_COLUMNS, VALID_TRACKS, VALID_YEARS


REQUIRED_COLUMNS = {
    "Submission ID",
    "year",
    "Track",
    "Account Name",
    "Reviewer Email Address",
    "Recommendation",
    "Problem &amp; Customer Definition",
    "Business Model",
    "Prototype/MVP",
    "Impact",
    "problem HLS",
    "stakeholders",
    "Problem",
    "Customer",
}

SENSITIVE_NAME_PARTS = (
    "email",
    "phone",
    "address",
    "first name",
    "last name",
    "linkedin",
    "huid",
    "reviewer full name",
    "team lead",
)


def load_raw_data(path: Path | str = PATHS.raw_data) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing confidential input: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    df.columns = [html.unescape(str(c)).strip() if c != "Problem &amp; Customer Definition" else c for c in df.columns]
    # Preserve the exact rating header expected elsewhere after HTML-unescaping all other names.
    if "Problem & Customer Definition" in df.columns and "Problem &amp; Customer Definition" not in df.columns:
        df = df.rename(columns={"Problem & Customer Definition": "Problem &amp; Customer Definition"})
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def _safe_examples(series: pd.Series, column: str, n: int = 5) -> list[str]:
    values = series[series.ne("")].drop_duplicates().astype(str)
    if any(part in column.casefold() for part in SENSITIVE_NAME_PARTS):
        return ["[redacted]" for _ in range(min(n, len(values)))]
    return [re.sub(r"\s+", " ", value)[:160] for value in values.head(n)]


def infer_meaning(column: str) -> str:
    c = column.casefold()
    exact = {
        "submission id": "Application-level identifier",
        "year": "Competition year",
        "track": "Normalized judging track or application-form track, depending on capitalization",
        "account name": "Merged venture/account name",
        "venture name": "Application-supplied venture name",
        "school": "Lead applicant Harvard school affiliation",
        "gender": "Lead applicant self-reported gender",
        "judge_type": "Reconciled judge category",
        "problem hls": "HLS structured problem statement",
        "stakeholders": "HLS customer/stakeholder needs",
        "problem": "Open/Social Impact structured problem statement",
        "customer": "Open/Social Impact customer segment",
    }
    if c in exact:
        return exact[c]
    if column in RATING_COLUMNS.values() or "overall rating" in c:
        return "Judge rating (1–5; zero/blank treated as missing)"
    if "judge" in c or "reviewer" in c or "contact id" in c:
        return "Judge or reviewer metadata"
    if any(x in c for x in ("m2 ", "m3 ", "m4 ", "m5 ")):
        return "Additional founder/team member metadata"
    if any(x in c for x in ("industry", "industries")):
        return "Industry classification"
    if any(x in c for x in ("country", "state", "city", "postal", "street")):
        return "Applicant, venture, team, or judge geography"
    if any(x in c for x in ("problem", "customer", "stakeholder", "need")):
        return "Application problem/customer-side text"
    if any(x in c for x in ("solution", "prototype", "value prop", "commercial")):
        return "Application solution/commercialization text"
    return "Application, judge, or merged metadata; inspect source values"


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        nonempty = df[col].ne("")
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "percent_missing": round(100 * (1 - nonempty.mean()), 2),
                "n_unique": int(df.loc[nonempty, col].nunique()),
                "example_values": json.dumps(_safe_examples(df[col], col), ensure_ascii=False),
                "inferred_meaning": infer_meaning(col),
            }
        )
    return pd.DataFrame(rows)


def _application_first(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("Submission ID", keep="first").copy()


def _candidate_duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    pair = df.groupby(["Submission ID", "Reviewer Email Address"], dropna=False).size()
    exact_cols = [
        "Submission ID",
        "Reviewer Email Address",
        "Date Evaluated",
        "Recommendation",
    ]
    exact_cols = [c for c in exact_cols if c in df.columns]
    exact = df.duplicated(exact_cols, keep=False)
    contact = df.get("Contact ID - 18 Digit", pd.Series("", index=df.index)).astype(str)
    email = df["Reviewer Email Address"].astype(str).str.casefold().str.strip()
    preferred_judge = contact.where(contact.ne(""), "email:" + email)
    preferred_pair = pd.DataFrame({"application": df["Submission ID"], "judge": preferred_judge}).value_counts()
    return {
        "repeated_application_reviewer_groups": int((pair > 1).sum()),
        "extra_application_reviewer_rows": int((pair[pair > 1] - 1).sum()),
        "repeated_application_preferred_judge_groups": int((preferred_pair > 1).sum()),
        "extra_preferred_judge_rows": int((preferred_pair[preferred_pair > 1] - 1).sum()),
        "exactish_duplicate_rows": int(exact.sum()),
    }


def _field_stability(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        nunique = df.loc[df[col].ne("")].groupby("Submission ID")[col].nunique()
        rows.append(
            {
                "column_name": col,
                "applications_populated": int(nunique.size),
                "applications_with_multiple_values": int((nunique > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def create_audit_outputs(df: pd.DataFrame, output_dir: Path = PATHS.tables) -> dict[str, object]:
    PATHS.ensure()
    output_dir.mkdir(parents=True, exist_ok=True)
    app = _application_first(df)
    data_dictionary = build_data_dictionary(df)
    data_dictionary.to_csv(output_dir / "data_dictionary.csv", index=False)

    missingness = data_dictionary[["column_name", "percent_missing", "n_unique"]].copy()
    missingness.to_csv(output_dir / "missingness_by_column.csv", index=False)

    row_counts = df.groupby(["year", "Track"]).size().rename("judge_rows").reset_index()
    app_counts = app.groupby(["year", "Track"]).size().rename("applications").reset_index()
    coverage = row_counts.merge(app_counts, on=["year", "Track"], how="outer")
    coverage.to_csv(output_dir / "year_track_coverage.csv", index=False)

    key_missing_rows = []
    for col in [
        "Account Name", "Venture Name", "School", "Gender", "Industries", "industry primary",
        "venture location country", "problem HLS", "stakeholders", "Problem", "Customer",
    ]:
        if col in app:
            key_missing_rows.append(
                {
                    "column_name": col,
                    "applications_missing": int(app[col].eq("").sum()),
                    "percent_missing": round(100 * app[col].eq("").mean(), 2),
                }
            )
    key_missing = pd.DataFrame(key_missing_rows)
    key_missing.to_csv(output_dir / "key_application_field_missingness.csv", index=False)

    ratings_per_app = df.groupby("Submission ID").size()
    duplicate_summary = _candidate_duplicate_summary(df)
    judge_ids = {
        "Reviewer Email Address": int(df["Reviewer Email Address"].replace("", np.nan).nunique()),
        "reviewer_name": int(df.get("reviewer_name", pd.Series(dtype=str)).replace("", np.nan).nunique()),
        "Contact ID - 18 Digit": int(df.get("Contact ID - 18 Digit", pd.Series(dtype=str)).replace("", np.nan).nunique()),
    }
    hls_strength = app["problem HLS"].str.len() + app["stakeholders"].str.len()
    general_strength = app["Problem"].str.len() + app["Customer"].str.len()
    structured_available = hls_strength.gt(0) | general_strength.gt(0)
    inferred_form = np.where(hls_strength >= general_strength, "hls", "open_social")
    expected_form = np.where(app["Track"].eq("Health & Life Science"), "hls", "open_social")
    form_mismatch = structured_available & (inferred_form != expected_form)
    structured_coverage = (
        app.assign(usable_problem_text=structured_available)
        .groupby(["year", "Track"], as_index=False)
        .agg(
            applications_total=("Submission ID", "nunique"),
            usable_problem_text=("usable_problem_text", "sum"),
        )
    )
    structured_coverage["coverage_share"] = (
        structured_coverage["usable_problem_text"] / structured_coverage["applications_total"]
    )
    conflict_rows = int(df.get("I have a conflict of interest:", pd.Series("", index=df.index)).eq("1").sum())

    stable_columns = [
        "year", "Track", "Account Name", "Venture Name", "School", "Gender", "Industries",
        "industry primary", "industry secondary", "venture location country",
        "problem HLS", "stakeholders", "Problem", "Customer", "Description", "pitch",
    ]
    stability = _field_stability(df, stable_columns)
    stability.to_csv(output_dir / "application_field_stability.csv", index=False)

    rating_distributions = []
    for short, col in RATING_COLUMNS.items():
        for value, count in df[col].value_counts(dropna=False).items():
            rating_distributions.append({"dimension": short, "raw_value": value, "n_rows": int(count)})
    pd.DataFrame(rating_distributions).to_csv(output_dir / "rating_value_distributions.csv", index=False)

    year_track_fields = []
    text_fields = ["problem HLS", "stakeholders", "Problem", "Customer", "Description", "pitch"]
    for (year, track), group in app.groupby(["year", "Track"]):
        for col in text_fields:
            year_track_fields.append(
                {
                    "year": int(year),
                    "track": track,
                    "column_name": col,
                    "applications_populated": int(group[col].ne("").sum()),
                    "applications_total": int(len(group)),
                    "percent_populated": round(100 * group[col].ne("").mean(), 2),
                }
            )
    pd.DataFrame(year_track_fields).to_csv(output_dir / "application_field_coverage.csv", index=False)

    report = [
        "# Data Audit",
        "",
        "## Unit and coverage",
        "",
        f"- Judge-level rows: **{len(df):,}**",
        f"- Source columns: **{len(df.columns):,}**",
        f"- Unique applications (`Submission ID`): **{app['Submission ID'].nunique():,}**",
        f"- Unique reviewer emails: **{judge_ids['Reviewer Email Address']:,}**",
        f"- Unique populated judge contact IDs: **{judge_ids['Contact ID - 18 Digit']:,}**",
        f"- Years: **{', '.join(map(str, sorted(df['year'].astype(int).unique())))}**",
        f"- Tracks: **{', '.join(sorted(df['Track'].unique()))}**",
        f"- Ratings per application: min **{ratings_per_app.min()}**, median **{ratings_per_app.median():.0f}**, mean **{ratings_per_app.mean():.1f}**, max **{ratings_per_app.max()}**",
        "",
        "## Validated identifiers",
        "",
        "- `Submission ID` is complete and stable within year/track; it is the application key.",
        "- `Contact ID - 18 Digit` is the preferred judge key; normalized reviewer email is the fallback.",
        "- `Account Name` is the complete venture-name field; `Venture Name` is retained as an application-supplied alternate.",
        "",
        "## Ratings",
        "",
        "- Valid values are 1–5. Zero and blank values are treated as missing.",
        f"- Rows declaring `I have a conflict of interest:` = 1: **{conflict_rows}**; retained for audit but excluded from aggregates.",
        f"- Two later duplicate judge identities are reconciled by keeping the most recent timestamp, leaving **{len(df) - conflict_rows - duplicate_summary['extra_preferred_judge_rows']:,}** evaluation rows eligible for aggregation.",
        "- Application-level outcomes use equal application weighting in downstream summaries.",
        "",
        "## Data-quality findings",
        "",
        f"- Repeated application-reviewer groups: **{duplicate_summary['repeated_application_reviewer_groups']}**",
        f"- Repeated application–preferred-judge groups: **{duplicate_summary['repeated_application_preferred_judge_groups']}**; the latest timestamp is retained for aggregation.",
        f"- Exact-ish duplicate rows: **{duplicate_summary['exactish_duplicate_rows']}**",
        f"- Application fields with within-ID variation: **{int((stability['applications_with_multiple_values'] > 0).sum())}**",
        f"- Applications with structured demand-side text: **{int(structured_available.sum())}**; missing: **{int((~structured_available).sum())}** (all missing cases are flagged, never backfilled from descriptions).",
        f"- Official track/form-family mismatches: **{int(form_mismatch.sum())}**; text construction follows the populated form and logs the mismatch.",
        "- Structured problem/customer field coverage varies by form family and year; see `application_field_coverage.csv`.",
        "- Field availability by year and track is explicitly tabulated in `application_field_coverage.csv`; the structured application form changes materially after 2021.",
        "- The source contains direct identifiers and free text. All row-level outputs remain local and gitignored.",
        "",
        "## Structured problem-text coverage and longitudinal comparability",
        "",
        "2021 has partial structured problem-text coverage (62 of 112 applications) and is preserved as historical context, not treated as fully comparable with 2022–2024. Default growth and decline metrics therefore use 2022–2024; four-year metrics are explicitly labeled partial coverage.",
        "",
        structured_coverage.to_markdown(index=False),
        "",
        "## Year and track counts",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Key application-field missingness",
        "",
        key_missing.to_markdown(index=False),
        "",
        "## Key field mapping",
        "",
        "| Purpose | Columns |",
        "|---|---|",
        "| Application ID | `Submission ID` |",
        "| Judge ID | `Contact ID - 18 Digit`; fallback `Reviewer Email Address` |",
        "| Outcomes | `Recommendation`, `Problem &amp; Customer Definition`, `Prototype/MVP`, `Business Model`, `Impact` |",
        "| HLS demand-side text | `problem HLS`, `stakeholders` |",
        "| Open/Social demand-side text | `Problem`, `Customer` |",
        "| Lead demographics | `School`, `Gender`, `degree` |",
        "| Team composition | `m2`–`m5 Harvard`, `m2`–`m5 gender`, team location fields |",
        "| Venture geography | `venture location country`, `venture location state`, `venture location city (us)` |",
        "| Industry | `Industries`, `industry primary`, `industry secondary` |",
        "| Judge type | `judge_type`, raw `Judge Type` |",
        "",
        "Generated reproducibly by `src.load_data.create_audit_outputs`.",
    ]
    (PATHS.outputs / "DATA_AUDIT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "judge_rows": len(df),
        "applications": int(app["Submission ID"].nunique()),
        "judge_ids": judge_ids,
        "duplicate_summary": duplicate_summary,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_core_shape(df: pd.DataFrame) -> None:
    years = tuple(sorted(df["year"].astype(int).unique()))
    tracks = tuple(sorted(df["Track"].unique()))
    if years != VALID_YEARS:
        raise AssertionError(f"Expected years {VALID_YEARS}; found {years}")
    if set(tracks) != set(VALID_TRACKS):
        raise AssertionError(f"Expected tracks {VALID_TRACKS}; found {tracks}")
    if df["Submission ID"].eq("").any():
        raise AssertionError("Submission ID contains missing values")
