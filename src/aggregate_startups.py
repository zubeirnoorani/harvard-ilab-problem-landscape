from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATHS, RATING_COLUMNS


APPLICATION_COLUMNS = [
    "Submission ID", "year", "Track", "Account Name", "Venture Name", "Description", "pitch",
    "Status", "Tag List", "Program", "PIC Track", "track", "Industries", "industry primary",
    "industry secondary", "venture location country", "venture location state",
    "venture location city (us)", "School", "Gender", "degree", "Grad Year",
    "location country", "Location State", "Location City us", "team_lead_nationality",
    "problem HLS", "stakeholders", "commercialization", "Market size", "Problem", "Customer",
    "Value prop", "Customer research",
    "m2 add", "m2 Harvard", "m2 otherschool", "m2 gender", "m2 location country",
    "m3 add", "m3 Harvard", "m3 otherschool", "m3 gender", "m3 location country",
    "m4 add", "m4 Harvard", "m4 otherschool", "m4 gender", "m4 location country",
    "m5 add", "m5 Harvard", "m5 otherschool", "m5 gender", "m5 location country",
]

JUDGE_OUTPUT_COLUMNS = [
    "Submission ID", "year", "Track", "Reviewer Email Address", "Contact ID - 18 Digit",
    "reviewer_name", "judge_type", "Judge Type", "Date Evaluated", "I have a conflict of interest:",
] + list(RATING_COLUMNS.values())


def normalize_venture_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def coerce_rating(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.between(1, 5))


def validate_application_stability(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for col in [c for c in APPLICATION_COLUMNS if c in df.columns and c != "Submission ID"]:
        nonempty = df.loc[df[col].ne(""), ["Submission ID", col]]
        varying = nonempty.groupby("Submission ID")[col].nunique()
        for application_id in varying[varying > 1].index:
            values = sorted(nonempty.loc[nonempty["Submission ID"].eq(application_id), col].unique())
            issues.append(
                {
                    "application_id": application_id,
                    "column_name": col,
                    "n_values": len(values),
                    "values": " | ".join(values)[:1000],
                }
            )
    return pd.DataFrame(issues, columns=["application_id", "column_name", "n_values", "values"])


def _first_nonempty(series: pd.Series) -> str:
    values = series[series.ne("")]
    return "" if values.empty else str(values.iloc[0])


def build_startup_level(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issues = validate_application_stability(df)
    stable_issue_cols = set(issues["column_name"]) if not issues.empty else set()
    if stable_issue_cols.intersection({"year", "Track", "Account Name"}):
        raise ValueError(f"Core application fields vary within ID: {sorted(stable_issue_cols)}")

    available_app_cols = [c for c in APPLICATION_COLUMNS if c in df.columns]
    applications = (
        df.groupby("Submission ID", sort=False, as_index=False)[available_app_cols[1:]]
        .agg(_first_nonempty)
        .rename(columns={"Submission ID": "application_id"})
    )
    applications["year"] = applications["year"].astype(int)
    applications["venture_name"] = applications["Account Name"].where(
        applications["Account Name"].ne(""), applications.get("Venture Name", "")
    )
    applications["venture_name_normalized"] = applications["venture_name"].map(normalize_venture_name)
    year_count = applications.groupby("venture_name_normalized")["year"].transform("nunique")
    applications["returning_venture"] = year_count.gt(1)
    applications["venture_application_sequence"] = (
        applications.sort_values(["venture_name_normalized", "year", "application_id"])
        .groupby("venture_name_normalized")
        .cumcount()
        .add(1)
        .reindex(applications.index)
        .astype(int)
    )

    judge = df[[c for c in JUDGE_OUTPUT_COLUMNS if c in df.columns]].copy()
    judge = judge.rename(columns={"Submission ID": "application_id", "Track": "track_official"})
    judge["conflict_of_interest"] = judge["I have a conflict of interest:"].eq("1")
    contact = judge.get("Contact ID - 18 Digit", pd.Series("", index=judge.index)).astype(str)
    email = judge["Reviewer Email Address"].astype(str).str.casefold().str.strip()
    judge["judge_id"] = contact.where(contact.ne(""), "email:" + email)
    judge["evaluation_timestamp"] = pd.to_datetime(judge.get("Date Evaluated"), errors="coerce", utc=True)
    judge["duplicate_application_judge"] = judge.duplicated(["application_id", "judge_id"], keep=False)
    for short, source in RATING_COLUMNS.items():
        judge[short] = coerce_rating(judge[source])

    judge["included_in_aggregation"] = False
    candidate = judge.loc[~judge["conflict_of_interest"]].copy()
    candidate["_source_row"] = candidate.index
    ordered = candidate.sort_values(
        ["application_id", "judge_id", "evaluation_timestamp", "_source_row"],
        na_position="first",
        kind="stable",
    )
    retained_indices = ordered.loc[
        ~ordered.duplicated(["application_id", "judge_id"], keep="last")
    ].index
    judge.loc[retained_indices, "included_in_aggregation"] = True
    valid = judge.loc[judge["included_in_aggregation"]].copy()
    stats = []
    for short in RATING_COLUMNS:
        grouped = valid.groupby("application_id")[short]
        block = grouped.agg(["mean", "std", "median", "count"]).rename(
            columns={
                "mean": f"{short}_mean",
                "std": f"{short}_sd",
                "median": f"{short}_median",
                "count": f"{short}_n_judges",
            }
        )
        stats.append(block)
    outcomes = pd.concat(stats, axis=1).reset_index()
    judge_counts = valid.groupby("application_id")["judge_id"].nunique().rename("n_unique_judges").reset_index()
    startup = applications.merge(outcomes, on="application_id", how="left").merge(judge_counts, on="application_id", how="left")
    if startup["application_id"].duplicated().any():
        raise AssertionError("Startup-level output is not one row per application")
    return startup, judge, issues


def write_startup_outputs(
    startup: pd.DataFrame,
    judge: pd.DataFrame,
    issues: pd.DataFrame,
    output_dir: Path = PATHS.tables,
) -> None:
    PATHS.ensure()
    startup.to_csv(output_dir / "startup_level.csv", index=False)
    judge.to_csv(output_dir / "judge_level_clean.csv", index=False)
    issues.to_csv(output_dir / "application_reconciliation_issues.csv", index=False)
