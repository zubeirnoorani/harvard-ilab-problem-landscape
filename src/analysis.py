from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as smf

from .config import PATHS, RANDOM_SEED, RATING_COLUMNS, VALID_YEARS


DIMENSIONS = list(RATING_COLUMNS)
DEFAULT_TREND_YEARS = (2022, 2023, 2024)
BOOTSTRAP_ITERATIONS = 2_000


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    seed: int = RANDOM_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    """Application-level percentile bootstrap interval for a mean."""
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(iterations, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_difference_ci(
    first: pd.Series | np.ndarray,
    second: pd.Series | np.ndarray,
    *,
    seed: int = RANDOM_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a difference in application-level means."""
    first_clean = pd.to_numeric(pd.Series(first), errors="coerce").dropna().to_numpy(dtype=float)
    second_clean = pd.to_numeric(pd.Series(second), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(first_clean) or not len(second_clean):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    first_samples = rng.choice(first_clean, size=(iterations, len(first_clean)), replace=True).mean(axis=1)
    second_samples = rng.choice(second_clean, size=(iterations, len(second_clean)), replace=True).mean(axis=1)
    low, high = np.quantile(first_samples - second_samples, [0.025, 0.975])
    return float(low), float(high)


def concept_quality_badge(stability: pd.Series, diagnostic: pd.Series) -> str:
    """Translate recorded stability and human review evidence into a visible badge."""
    notes = str(diagnostic.get("problem_quality_notes", ""))
    caution = bool(
        re.search(
            r"geography-driven|heterogeneous|mixed|conflates|not leadership-ready|"
            r"use cautiously|solution/technology|drift toward solution",
            notes,
            flags=re.IGNORECASE,
        )
    )
    if bool(stability.get("small_and_unstable", False)) or bool(stability.get("small_primary_area", False)):
        return "Small / exploratory"
    if caution or str(stability.get("stability_flag", "")).lower() == "low":
        return "Mixed / interpret cautiously"
    if re.search(r"broad|overlap|broaden", notes, flags=re.IGNORECASE) or str(stability.get("stability_flag", "")).lower() == "moderate":
        return "Stable but broad"
    return "Stable"


def load_analysis_inputs(m_concepts: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    startup = pd.read_csv(PATHS.tables / "startup_level.csv", low_memory=False)
    judge = pd.read_csv(PATHS.tables / "judge_level_clean.csv", low_memory=False)
    assignments = pd.read_csv(PATHS.tables / f"concept_assignments_m{m_concepts}.csv", low_memory=False)
    diagnostics = pd.read_csv(PATHS.tables / f"concept_diagnostics_m{m_concepts}.csv", low_memory=False)
    return startup, judge, assignments, diagnostics


def _summarize_outcomes(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_applications"] = int(group["application_id"].nunique())
        row["n_startups"] = row["n_applications"]  # Backward-compatible analytical column.
        row["share_of_scope"] = np.nan
        for dimension in DIMENSIONS:
            values = pd.to_numeric(group[f"{dimension}_mean"], errors="coerce")
            row[f"{dimension}_mean"] = float(values.mean())
            row[f"{dimension}_sd_between_startups"] = float(values.std(ddof=1))
            row[f"{dimension}_median"] = float(values.median())
            row[f"{dimension}_n_startups"] = int(values.notna().sum())
            row[f"{dimension}_n_ratings"] = int(
                pd.to_numeric(group[f"{dimension}_n_judges"], errors="coerce").fillna(0).sum()
            )
        raw_low, raw_high = bootstrap_mean_ci(group["recommendation_mean"])
        adjusted = pd.to_numeric(group["recommendation_adjusted_year_track"], errors="coerce")
        adjusted_low, adjusted_high = bootstrap_mean_ci(adjusted)
        row["recommendation_ci_low"] = raw_low
        row["recommendation_ci_high"] = raw_high
        row["recommendation_adjusted_year_track_mean"] = float(adjusted.mean())
        row["recommendation_adjusted_year_track_sd_between_applications"] = float(adjusted.std(ddof=1))
        row["recommendation_adjusted_year_track_median"] = float(adjusted.median())
        row["recommendation_adjusted_year_track_n_applications"] = int(adjusted.notna().sum())
        row["recommendation_adjusted_year_track_ci_low"] = adjusted_low
        row["recommendation_adjusted_year_track_ci_high"] = adjusted_high
        rows.append(row)
    return pd.DataFrame(rows)


def build_membership_long(
    startup: pd.DataFrame,
    assignments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    m_concepts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_columns = [c for c in assignments.columns if c != "application_id" and c in startup.columns]
    base = assignments.merge(
        startup.drop(columns=duplicate_columns),
        on="application_id",
        how="left",
        validate="one_to_one",
    )
    label_map = diagnostics.set_index("concept_id")["concept_label"].to_dict()
    activation_cols = [f"concept_{i:02d}_activation" for i in range(m_concepts)]
    long = base.melt(
        id_vars=[c for c in base.columns if c not in activation_cols],
        value_vars=activation_cols,
        var_name="activation_column",
        value_name="concept_activation",
    )
    long["concept_id"] = long["activation_column"].str.extract(r"concept_(\d+)_").astype(int)
    long = long.loc[long["concept_activation"] > 0].copy()
    long["concept_label"] = long["concept_id"].map(label_map)
    long["membership_type"] = "active_feature"

    primary = base.loc[base["primary_problem_concept"] >= 0].copy()
    primary["concept_id"] = primary["primary_problem_concept"].astype(int)
    primary["concept_label"] = primary["primary_problem_label"]
    primary["concept_activation"] = primary["primary_activation"]
    primary["membership_type"] = "primary_assignment"
    return long, primary


def summarize_concept_outcomes(
    membership: pd.DataFrame,
    membership_type: str,
    m_concepts: int,
) -> pd.DataFrame:
    blocks = []
    scopes = [([], "All tracks"), (["Track"], "Track")]
    for scope_cols, scope_name in scopes:
        for time_cols, period in [([], "All years"), (["year"], "Year")]:
            groups = scope_cols + time_cols + ["concept_id", "concept_label"]
            summary = _summarize_outcomes(membership, groups)
            summary["scope"] = scope_name
            summary["period"] = period
            summary["membership_type"] = membership_type
            denominator_cols = scope_cols + time_cols
            if denominator_cols:
                denominators = membership.drop_duplicates("application_id").groupby(denominator_cols)["application_id"].nunique()
                summary["scope_n_startups"] = [
                    int(denominators.loc[tuple(row[c] for c in denominator_cols)])
                    if len(denominator_cols) > 1
                    else int(denominators.loc[row[denominator_cols[0]]])
                    for _, row in summary.iterrows()
                ]
            else:
                summary["scope_n_startups"] = membership["application_id"].nunique()
            summary["share_of_scope"] = summary["n_startups"] / summary["scope_n_startups"]
            blocks.append(summary)
    out = pd.concat(blocks, ignore_index=True)
    out["m_concepts"] = m_concepts
    return out


def build_trends(primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    rows = []
    for track in ["All tracks"] + sorted(primary["Track"].dropna().unique().tolist()):
        scope = primary if track == "All tracks" else primary.loc[primary["Track"].eq(track)]
        for (concept_id, concept_label), group in scope.groupby(["concept_id", "concept_label"]):
            counts = group.groupby("year")["application_id"].nunique().reindex(VALID_YEARS, fill_value=0)
            denominators = scope.groupby("year")["application_id"].nunique().reindex(VALID_YEARS, fill_value=0)
            shares = counts.div(denominators.replace(0, np.nan))
            valid_default = shares.loc[list(DEFAULT_TREND_YEARS)].notna()
            default_years = np.asarray(DEFAULT_TREND_YEARS)[valid_default]
            default_shares = shares.loc[list(DEFAULT_TREND_YEARS)][valid_default]
            slope_default = linregress(default_years, default_shares).slope if valid_default.sum() >= 2 else np.nan
            valid_partial = shares.notna()
            slope_partial = (
                linregress(np.asarray(VALID_YEARS)[valid_partial], shares[valid_partial]).slope
                if valid_partial.sum() >= 2
                else np.nan
            )
            row = {
                "track_scope": track,
                "concept_id": int(concept_id),
                "concept_label": concept_label,
                "trend_slope_2022_2024": float(slope_default),
                "change_2022_to_2024": float(shares.loc[2024] - shares.loc[2022])
                if pd.notna(shares.loc[2022]) and pd.notna(shares.loc[2024])
                else np.nan,
                "trend_slope_2021_2024_partial": float(slope_partial),
                "change_2021_to_2024_partial": float(shares.loc[2024] - shares.loc[2021])
                if pd.notna(shares.loc[2021]) and pd.notna(shares.loc[2024])
                else np.nan,
                "default_trend_period": "2022–2024",
                "coverage_2021": "Partial problem-text coverage",
                "small_n_flag": bool(counts.max() < 10),
                "m_concepts": m_concepts,
            }
            for year in VALID_YEARS:
                row[f"count_{year}"] = int(counts.loc[year])
                row[f"share_{year}"] = float(shares.loc[year]) if pd.notna(shares.loc[year]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_disagreement(primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    rows = []
    for track in ["All tracks"] + sorted(primary["Track"].dropna().unique().tolist()):
        scope = primary if track == "All tracks" else primary.loc[primary["Track"].eq(track)]
        grouped = scope.groupby(["concept_id", "concept_label"], dropna=False)
        for (concept_id, concept_label), group in grouped:
            rating = pd.to_numeric(group["recommendation_mean"], errors="coerce")
            disagreement = pd.to_numeric(group["recommendation_sd"], errors="coerce")
            rows.append(
                {
                    "track_scope": track,
                    "concept_id": int(concept_id),
                    "concept_label": concept_label,
                    "recommendation_mean": float(rating.mean()),
                    "mean_within_startup_recommendation_sd": float(disagreement.mean()),
                    "median_within_startup_recommendation_sd": float(disagreement.median()),
                    "n_startups": int(group["application_id"].nunique()),
                    "n_underlying_ratings": int(pd.to_numeric(group["recommendation_n_judges"], errors="coerce").sum()),
                    "m_concepts": m_concepts,
                }
            )
    out = pd.DataFrame(rows)
    classifications = []
    for _, scope in out.groupby("track_scope"):
        rating_cut = scope["recommendation_mean"].median()
        disagreement_cut = scope["mean_within_startup_recommendation_sd"].median()
        for idx, row in scope.iterrows():
            classifications.append(
                (
                    idx,
                    ("High rating" if row["recommendation_mean"] >= rating_cut else "Low rating")
                    + " / "
                    + ("High disagreement" if row["mean_within_startup_recommendation_sd"] >= disagreement_cut else "Low disagreement"),
                )
            )
    class_map = dict(classifications)
    out["rating_disagreement_quadrant"] = out.index.map(class_map)
    out["small_n_flag"] = out["n_startups"] < 10
    return out


def analyze_judge_type(primary: pd.DataFrame, judge: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    keys = primary[["application_id", "year", "Track", "concept_id", "concept_label"]]
    if "included_in_aggregation" in judge:
        valid_judges = judge.loc[judge["included_in_aggregation"].astype(bool)]
    else:
        valid_judges = judge.loc[~judge["conflict_of_interest"].astype(bool)]
    merged = valid_judges.merge(keys, on="application_id", how="inner")
    merged["recommendation"] = pd.to_numeric(merged["recommendation"], errors="coerce")
    merged["year_track_mean"] = merged.groupby(["year_x", "track_official"])["recommendation"].transform("mean")
    merged["recommendation_centered_year_track"] = merged["recommendation"] - merged["year_track_mean"]
    result = (
        merged.groupby(["concept_id", "concept_label", "judge_type"], dropna=False)
        .agg(
            recommendation_mean=("recommendation", "mean"),
            recommendation_sd=("recommendation", "std"),
            recommendation_median=("recommendation", "median"),
            n_ratings=("recommendation", "count"),
            n_startups=("application_id", "nunique"),
            mean_year_track_centered_rating=("recommendation_centered_year_track", "mean"),
        )
        .reset_index()
    )
    result["m_concepts"] = m_concepts
    result["small_n_flag"] = result["n_ratings"] < 20
    return result


def build_founder_long(startup: pd.DataFrame, primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    concept_keys = primary[["application_id", "concept_id", "concept_label"]]
    base = startup.merge(concept_keys, on="application_id", how="inner")
    def supplied(value: object) -> str:
        if pd.isna(value) or not str(value).strip():
            return "Missing"
        return str(value).strip()

    rows = []
    for _, row in base.iterrows():
        rows.append(
            {
                "application_id": row["application_id"], "year": row["year"], "Track": row["Track"],
                "concept_id": row["concept_id"], "concept_label": row["concept_label"],
                "member_number": 1, "gender": supplied(row.get("Gender", "")),
                "harvard_school": supplied(row.get("School", "")),
                "country": supplied(row.get("location country", "")),
            }
        )
        for member in range(2, 6):
            values = [row.get(f"m{member} Harvard", ""), row.get(f"m{member} gender", ""), row.get(f"m{member} location country", "")]
            if not any(pd.notna(value) and str(value).strip() for value in values):
                continue
            rows.append(
                {
                    "application_id": row["application_id"], "year": row["year"], "Track": row["Track"],
                    "concept_id": row["concept_id"], "concept_label": row["concept_label"],
                    "member_number": member,
                    "gender": supplied(row.get(f"m{member} gender", "")),
                    "harvard_school": supplied(row.get(f"m{member} Harvard", "")),
                    "country": supplied(row.get(f"m{member} location country", "")),
                }
            )
    founder = pd.DataFrame(rows)
    founder["m_concepts"] = m_concepts
    return founder


def composition_tables(
    primary: pd.DataFrame,
    founder: pd.DataFrame,
    m_concepts: int,
) -> dict[str, pd.DataFrame]:
    outputs = {}
    for field in ("gender", "harvard_school", "country"):
        table = (
            founder.groupby(["concept_id", "concept_label", field], dropna=False)
            .size().rename("n_founders").reset_index()
        )
        totals = table.groupby("concept_id")["n_founders"].transform("sum")
        table["share_within_problem"] = table["n_founders"] / totals
        table["m_concepts"] = m_concepts
        outputs[field] = table
    for field in ("Track", "year", "venture location country", "industry primary"):
        table = (
            primary.groupby(["concept_id", "concept_label", field], dropna=False)
            .size().rename("n_startups").reset_index()
        )
        totals = table.groupby("concept_id")["n_startups"].transform("sum")
        table["share_within_problem"] = table["n_startups"] / totals
        table["m_concepts"] = m_concepts
        outputs[field] = table
    school_matrix = outputs["harvard_school"].pivot_table(
        index="harvard_school", columns="concept_label", values="share_within_problem", fill_value=0
    )
    outputs["school_matrix"] = school_matrix
    return outputs


def build_track_overrepresentation(primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    """Compare each track's concept concentration with the full usable portfolio."""
    overall_n = int(primary["application_id"].nunique())
    overall_counts = primary.groupby(["concept_id", "concept_label"])["application_id"].nunique()
    rows: list[dict[str, object]] = []
    for track, track_frame in primary.groupby("Track"):
        track_n = int(track_frame["application_id"].nunique())
        track_counts = track_frame.groupby(["concept_id", "concept_label"])["application_id"].nunique()
        for (concept_id, concept_label), overall_concept_n in overall_counts.items():
            n = int(track_counts.get((concept_id, concept_label), 0))
            track_share = n / track_n if track_n else np.nan
            overall_share = int(overall_concept_n) / overall_n if overall_n else np.nan
            rows.append(
                {
                    "Track": track,
                    "concept_id": int(concept_id),
                    "concept_label": concept_label,
                    "n_applications": n,
                    "track_n_applications": track_n,
                    "overall_concept_n_applications": int(overall_concept_n),
                    "overall_n_applications": overall_n,
                    "track_share": track_share,
                    "overall_share": overall_share,
                    "overrepresentation_ratio": track_share / overall_share if overall_share else np.nan,
                    "share_difference": track_share - overall_share,
                    "small_n_flag": n < 10,
                    "m_concepts": m_concepts,
                }
            )
    return pd.DataFrame(rows)


def lead_gender_composition(primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    """Application-level lead gender, distinct from team-member records."""
    data = primary[["application_id", "concept_id", "concept_label", "Gender"]].copy()
    data["lead_gender"] = data["Gender"].where(data["Gender"].notna() & data["Gender"].astype(str).str.strip().ne(""), "Missing")
    rows = (
        data.groupby(["concept_id", "concept_label", "lead_gender"], dropna=False)["application_id"]
        .nunique()
        .rename("n_applications")
        .reset_index()
    )
    total = data.groupby("concept_id")["application_id"].nunique()
    reported = data.loc[data["lead_gender"].ne("Missing")].groupby("concept_id")["application_id"].nunique()
    rows["problem_n_applications"] = rows["concept_id"].map(total).astype(int)
    rows["reported_lead_gender_n"] = rows["concept_id"].map(reported).fillna(0).astype(int)
    rows["reporting_coverage"] = rows["reported_lead_gender_n"] / rows["problem_n_applications"]
    rows["share_among_reported"] = np.where(
        rows["lead_gender"].ne("Missing") & rows["reported_lead_gender_n"].gt(0),
        rows["n_applications"] / rows["reported_lead_gender_n"],
        np.nan,
    )
    rows["m_concepts"] = m_concepts
    return rows


def gender_problem_evaluation(
    primary: pd.DataFrame,
    *,
    min_cell: int = 10,
    seed: int = RANDOM_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Describe lead-gender evaluation patterns at the application level.

    The pooled coefficient adjusts for problem fixed effects and year-by-track
    fixed effects. Problem-specific rows are published only when both the
    female- and male-led cells meet ``min_cell``. No gender is inferred.
    """
    required = {
        "application_id",
        "year",
        "Track",
        "concept_id",
        "concept_label",
        "Gender",
        "recommendation_mean",
        "recommendation_adjusted_year_track",
    }
    missing = required.difference(primary.columns)
    if missing:
        raise ValueError(f"Missing columns for gender evaluation: {sorted(missing)}")

    data = primary[list(required)].copy()
    data["lead_gender"] = data["Gender"].fillna("").astype(str).str.strip()
    data["gender_key"] = data["lead_gender"].str.casefold()
    data = data.loc[data["gender_key"].isin({"female", "male"})].copy()
    data["female_lead"] = data["gender_key"].eq("female").astype(int)
    data["recommendation_mean"] = pd.to_numeric(data["recommendation_mean"], errors="coerce")
    data["recommendation_adjusted_year_track"] = pd.to_numeric(
        data["recommendation_adjusted_year_track"], errors="coerce"
    )
    data["year_track"] = data["year"].astype(str) + " × " + data["Track"].astype(str)

    model_data = data.dropna(subset=["recommendation_mean"]).copy()
    fitted = smf.ols(
        "recommendation_mean ~ female_lead + C(concept_id) + C(year_track)",
        data=model_data,
    ).fit(cov_type="HC3")
    coefficient = float(fitted.params["female_lead"])
    ci = fitted.conf_int().loc["female_lead"]

    all_gender = primary["Gender"].fillna("").astype(str).str.strip().str.casefold()
    female_total = int(all_gender.eq("female").sum())
    male_total = int(all_gender.eq("male").sum())
    total_applications = int(primary["application_id"].nunique())
    reported_total = int(all_gender.ne("").sum())

    overall_groups: dict[str, dict[str, float | int]] = {}
    for label, key in (("Female", "female"), ("Male", "male")):
        group = data.loc[data["gender_key"].eq(key)]
        raw_low, raw_high = bootstrap_mean_ci(
            group["recommendation_mean"], seed=seed + (1 if key == "female" else 2), iterations=iterations
        )
        adjusted_low, adjusted_high = bootstrap_mean_ci(
            group["recommendation_adjusted_year_track"],
            seed=seed + (3 if key == "female" else 4),
            iterations=iterations,
        )
        overall_groups[label] = {
            "n": int(group["application_id"].nunique()),
            "raw_mean": float(group["recommendation_mean"].mean()),
            "raw_ci_low": raw_low,
            "raw_ci_high": raw_high,
            "adjusted_mean": float(group["recommendation_adjusted_year_track"].mean()),
            "adjusted_ci_low": adjusted_low,
            "adjusted_ci_high": adjusted_high,
        }

    rows: list[dict[str, object]] = []
    for (concept_id, concept_label), group in data.groupby(["concept_id", "concept_label"], sort=False):
        female = group.loc[group["gender_key"].eq("female")]
        male = group.loc[group["gender_key"].eq("male")]
        female_n = int(female["application_id"].nunique())
        male_n = int(male["application_id"].nunique())
        publishable = female_n >= min_cell and male_n >= min_cell
        row: dict[str, object] = {
            "concept_id": int(concept_id),
            "concept_label": str(concept_label),
            "comparison_publishable": publishable,
        }
        if publishable:
            female_raw_ci = bootstrap_mean_ci(
                female["recommendation_mean"],
                seed=seed + 100 + int(concept_id) * 10,
                iterations=iterations,
            )
            male_raw_ci = bootstrap_mean_ci(
                male["recommendation_mean"],
                seed=seed + 101 + int(concept_id) * 10,
                iterations=iterations,
            )
            female_adjusted_ci = bootstrap_mean_ci(
                female["recommendation_adjusted_year_track"],
                seed=seed + 102 + int(concept_id) * 10,
                iterations=iterations,
            )
            male_adjusted_ci = bootstrap_mean_ci(
                male["recommendation_adjusted_year_track"],
                seed=seed + 103 + int(concept_id) * 10,
                iterations=iterations,
            )
            difference_ci = bootstrap_difference_ci(
                female["recommendation_adjusted_year_track"],
                male["recommendation_adjusted_year_track"],
                seed=seed + 104 + int(concept_id) * 10,
                iterations=iterations,
            )
            row.update(
                {
                    "female_n": female_n,
                    "male_n": male_n,
                    "female_raw_mean": float(female["recommendation_mean"].mean()),
                    "female_raw_ci_low": female_raw_ci[0],
                    "female_raw_ci_high": female_raw_ci[1],
                    "male_raw_mean": float(male["recommendation_mean"].mean()),
                    "male_raw_ci_low": male_raw_ci[0],
                    "male_raw_ci_high": male_raw_ci[1],
                    "female_adjusted_mean": float(female["recommendation_adjusted_year_track"].mean()),
                    "female_adjusted_ci_low": female_adjusted_ci[0],
                    "female_adjusted_ci_high": female_adjusted_ci[1],
                    "male_adjusted_mean": float(male["recommendation_adjusted_year_track"].mean()),
                    "male_adjusted_ci_low": male_adjusted_ci[0],
                    "male_adjusted_ci_high": male_adjusted_ci[1],
                    "adjusted_difference_female_minus_male": float(
                        female["recommendation_adjusted_year_track"].mean()
                        - male["recommendation_adjusted_year_track"].mean()
                    ),
                    "adjusted_difference_ci_low": difference_ci[0],
                    "adjusted_difference_ci_high": difference_ci[1],
                }
            )
        rows.append(row)

    summary: dict[str, object] = {
        "total_applications": total_applications,
        "reported_lead_gender_n": reported_total,
        "female_n": female_total,
        "male_n": male_total,
        "model_n": int(fitted.nobs),
        "female_coefficient": coefficient,
        "female_coefficient_ci_low": float(ci.iloc[0]),
        "female_coefficient_ci_high": float(ci.iloc[1]),
        "female_coefficient_p_value": float(fitted.pvalues["female_lead"]),
        "model": "Recommendation ~ female lead + problem fixed effects + year×track fixed effects",
        "covariance": "HC3 heteroskedasticity-robust",
        "overall_groups": overall_groups,
        "minimum_public_cell": min_cell,
    }
    return summary, pd.DataFrame(rows).sort_values("concept_id").reset_index(drop=True)


def write_gender_problem_evaluation_note(
    primary: pd.DataFrame,
    *,
    output_path: Path | None = None,
    min_cell: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Write the meeting backup note and its local analytical table."""
    summary, table = gender_problem_evaluation(primary, min_cell=min_cell)
    table.to_csv(PATHS.tables / "gender_problem_evaluation_m16.csv", index=False)
    output_path = output_path or (PATHS.outputs / "GENDER_PROBLEM_EVALUATION_NOTE.md")
    publishable = table.loc[table["comparison_publishable"].astype(bool)].copy()
    lines = [
        "# Lead-applicant gender × problem evaluation note",
        "",
        "## Meeting question",
        "",
        "Within the same underlying customer problems, do applications led by women appear to receive different Recommendation ratings? This is a descriptive application-level comparison, not evidence of a gender effect or any causal mechanism.",
        "",
        "## Coverage",
        "",
        f"- Applications with usable problem text: **{summary['total_applications']}**",
        f"- Female lead applicant: **N={summary['female_n']}**",
        f"- Male lead applicant: **N={summary['male_n']}**",
        f"- Any supplied lead-gender response: **N={summary['reported_lead_gender_n']}**",
        "- Other supplied response categories and nonresponse are not broken out because the category cells are below the public disclosure threshold.",
        "",
        "## Adjusted portfolio-wide result",
        "",
        (
            f"The female-lead coefficient is **{summary['female_coefficient']:+.2f} Recommendation points** "
            f"(HC3 95% CI **{summary['female_coefficient_ci_low']:+.2f} to "
            f"{summary['female_coefficient_ci_high']:+.2f}**, N={summary['model_n']} applications) in:"
        ),
        "",
        f"`{summary['model']}`",
        "",
        "The interval should be read as uncertainty around an observational association. It does not isolate a gender effect and may reflect applicant, judge, cohort, selection, or other composition differences.",
        "",
        "## Problem-specific descriptive comparisons",
        "",
        "Only concepts with at least 10 female-led and 10 male-led applications are shown. Adjusted values are application Recommendation centered within year × track; confidence intervals use a fixed-seed application-level percentile bootstrap.",
        "",
        "| Problem area | Female N | Male N | Female raw | Male raw | Female adjusted | Male adjusted | Adjusted difference (F−M) | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in publishable.sort_values("concept_label").iterrows():
        lines.append(
            f"| {row['concept_label']} | {int(row['female_n'])} | {int(row['male_n'])} | "
            f"{row['female_raw_mean']:.2f} | {row['male_raw_mean']:.2f} | "
            f"{row['female_adjusted_mean']:+.2f} | {row['male_adjusted_mean']:+.2f} | "
            f"{row['adjusted_difference_female_minus_male']:+.2f} | "
            f"{row['adjusted_difference_ci_low']:+.2f} to {row['adjusted_difference_ci_high']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- The supplied application-level Gender field is used; gender is never inferred from names.",
            "- The model adjusts for observed problem area and year × track, but not judge assignment, team composition, venture maturity, application quality, selection, or other unobserved differences.",
            "- Recommendation is a judging score, not a funding or venture-outcome measure.",
            "- Problem-specific comparisons are exploratory. Multiple comparisons and modest cell sizes make interaction estimates unstable.",
            "- Suppressed problem cells should be discussed qualitatively only after appropriate confidential review.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return summary, table


def team_gender_composition(founder: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    """Reported lead-and-team gender records with explicit coverage denominators."""
    data = founder.copy()
    data["gender"] = data["gender"].fillna("Missing").replace("", "Missing")
    rows = (
        data.groupby(["concept_id", "concept_label", "gender"], dropna=False)
        .size()
        .rename("n_founders")
        .reset_index()
    )
    total = data.groupby("concept_id").size()
    reported = data.loc[data["gender"].ne("Missing")].groupby("concept_id").size()
    rows["team_records_total"] = rows["concept_id"].map(total).astype(int)
    rows["team_gender_reported_n"] = rows["concept_id"].map(reported).fillna(0).astype(int)
    rows["reporting_coverage"] = rows["team_gender_reported_n"] / rows["team_records_total"]
    rows["share_among_reported"] = np.where(
        rows["gender"].ne("Missing") & rows["team_gender_reported_n"].gt(0),
        rows["n_founders"] / rows["team_gender_reported_n"],
        np.nan,
    )
    rows["m_concepts"] = m_concepts
    return rows


def school_representation_index(founder: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    """Normalize school presence in each problem by the reported-founder baseline."""
    data = founder.copy()
    data["harvard_school"] = data["harvard_school"].fillna("Missing").replace("", "Missing")
    reported = data.loc[data["harvard_school"].ne("Missing")].copy()
    portfolio_total = int(len(reported))
    portfolio_school = reported.groupby("harvard_school").size()
    problem_total = reported.groupby("concept_id").size()
    rows = (
        reported.groupby(["concept_id", "concept_label", "harvard_school"])
        .size()
        .rename("n_founders")
        .reset_index()
    )
    rows["reported_founders_in_problem"] = rows["concept_id"].map(problem_total).astype(int)
    rows["reported_founders_portfolio"] = portfolio_total
    rows["portfolio_school_n"] = rows["harvard_school"].map(portfolio_school).astype(int)
    rows["share_of_problem_from_school"] = rows["n_founders"] / rows["reported_founders_in_problem"]
    rows["share_of_all_reported_founders_from_school"] = rows["portfolio_school_n"] / portfolio_total
    rows["representation_index"] = (
        rows["share_of_problem_from_school"] / rows["share_of_all_reported_founders_from_school"]
    )
    rows["m_concepts"] = m_concepts
    return rows


def intervention_profiles(primary: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    columns = [f"{dimension}_mean" for dimension in ("problem_customer_definition", "solution_prototype", "business_model", "impact")]
    profile = primary.groupby(["concept_id", "concept_label"])[columns].agg(["mean", "count"])
    profile.columns = [f"{dimension}_{stat}" for dimension, stat in profile.columns]
    profile = profile.reset_index()
    profile["n_startups"] = profile["problem_customer_definition_mean_count"]
    mean_cols = [f"{column}_mean" for column in columns]
    for column in mean_cols:
        std = profile[column].std(ddof=0)
        profile[column.replace("_mean_mean", "_z")] = (profile[column] - profile[column].mean()) / std if std else 0
    profile["problem_minus_business"] = profile["problem_customer_definition_mean_mean"] - profile["business_model_mean_mean"]
    profile["problem_minus_prototype"] = profile["problem_customer_definition_mean_mean"] - profile["solution_prototype_mean_mean"]
    profile["prototype_minus_problem"] = profile["solution_prototype_mean_mean"] - profile["problem_customer_definition_mean_mean"]
    profile["impact_minus_business"] = profile["impact_mean_mean"] - profile["business_model_mean_mean"]

    def pattern(row: pd.Series) -> str:
        z_problem = row["problem_customer_definition_z"]
        z_solution = row["solution_prototype_z"]
        z_business = row["business_model_z"]
        z_impact = row["impact_z"]
        if min(z_problem, z_solution, z_business, z_impact) > 0:
            return "Relatively strong across dimensions"
        if z_problem > 0.35 and z_business < -0.35:
            return "Problem understood / business model gap"
        if z_solution > 0.35 and z_problem < -0.35:
            return "Prototype ahead of customer definition"
        if z_impact > 0.35 and z_business < -0.35:
            return "Strong impact / commercialization gap"
        return "Mixed profile"

    profile["support_pattern"] = profile.apply(pattern, axis=1)
    profile["m_concepts"] = m_concepts
    profile["small_n_flag"] = profile["n_startups"] < 10
    return profile


def strategic_map(primary: pd.DataFrame, trends: pd.DataFrame, disagreement: pd.DataFrame, m_concepts: int) -> pd.DataFrame:
    summary = _summarize_outcomes(primary, ["concept_id", "concept_label"])
    trend = trends.loc[trends["track_scope"].eq("All tracks")]
    disagree = disagreement.loc[disagreement["track_scope"].eq("All tracks"), [
        "concept_id", "mean_within_startup_recommendation_sd"
    ]]
    out = summary.merge(trend, on=["concept_id", "concept_label"], how="left").merge(disagree, on="concept_id", how="left")
    out["prevalence"] = out["n_startups"] / primary["application_id"].nunique()
    rating_cut = out["recommendation_adjusted_year_track_mean"].median()
    size_cut = out["n_startups"].median()
    growth_cut = 0.005
    disagree_cut = out["mean_within_startup_recommendation_sd"].median()

    def labels(row: pd.Series) -> str:
        values = []
        if row["recommendation_adjusted_year_track_mean"] >= rating_cut and row["trend_slope_2022_2024"] > growth_cut:
            values.append("relatively stronger evaluated + growing")
        if row["recommendation_adjusted_year_track_mean"] >= rating_cut and row["n_startups"] < size_cut:
            values.append("stronger evaluation + lower historical attention")
        if row["n_startups"] >= size_cut:
            values.append("highly populated")
        if row["trend_slope_2022_2024"] > growth_cut:
            values.append("emerging")
        if row["mean_within_startup_recommendation_sd"] >= disagree_cut:
            values.append("polarizing")
        return "; ".join(values) or "mixed"

    out["descriptive_category"] = out.apply(labels, axis=1)
    out["m_concepts"] = m_concepts
    out["small_n_flag"] = out["n_startups"] < 10
    return out


def problem_profile(
    concept_id: int,
    primary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trends: pd.DataFrame,
    disagreement: pd.DataFrame,
    intervention: pd.DataFrame,
    founder: pd.DataFrame,
    overlap: pd.DataFrame,
    stability: pd.DataFrame,
    track_overrepresentation: pd.DataFrame,
    m_concepts: int,
) -> dict[str, object]:
    group = primary.loc[primary["concept_id"].eq(concept_id)].copy()
    diagnostic = diagnostics.set_index("concept_id").loc[concept_id]
    trend = trends.loc[(trends["track_scope"].eq("All tracks")) & trends["concept_id"].eq(concept_id)].iloc[0]
    disagree = disagreement.loc[(disagreement["track_scope"].eq("All tracks")) & disagreement["concept_id"].eq(concept_id)].iloc[0]
    outcome = _summarize_outcomes(group, ["concept_id", "concept_label"]).iloc[0]
    stability_row = stability.set_index("concept_id").loc[concept_id]
    support_row = intervention.set_index("concept_id").loc[concept_id]
    dimensions = {dimension: float(group[f"{dimension}_mean"].mean()) for dimension in DIMENSIONS}

    def distribution(frame: pd.DataFrame, field: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in frame[field].fillna("Missing").replace("", "Missing").value_counts().items()}

    related = overlap.loc[(overlap["concept_a"].eq(concept_id)) | (overlap["concept_b"].eq(concept_id))].copy()
    related["related_concept_id"] = np.where(related["concept_a"].eq(concept_id), related["concept_b"], related["concept_a"])
    label_map = diagnostics.set_index("concept_id")["concept_label"].to_dict()
    related = related.nlargest(5, "jaccard")
    examples = json.loads(diagnostic["top_examples"])
    return {
        "m_concepts": m_concepts,
        "concept_id": concept_id,
        "problem_label": diagnostic["concept_label"],
        "description": diagnostic["description"],
        "quality_notes": diagnostic["problem_quality_notes"],
        "quality_badge": concept_quality_badge(stability_row, diagnostic),
        "activation_stability": str(stability_row["stability_flag"]),
        "membership_jaccard": float(stability_row["mean_matched_membership_jaccard"]),
        "activation_correlation": float(stability_row["mean_matched_activation_correlation"]),
        "n_applications": int(group["application_id"].nunique()),
        "share_of_portfolio": float(group["application_id"].nunique() / primary["application_id"].nunique()),
        "recommendation_mean": float(outcome["recommendation_mean"]),
        "recommendation_ci_95": [float(outcome["recommendation_ci_low"]), float(outcome["recommendation_ci_high"])],
        "recommendation_sd_between_applications": float(group["recommendation_mean"].std(ddof=1)),
        "recommendation_adjusted_year_track_mean": float(outcome["recommendation_adjusted_year_track_mean"]),
        "recommendation_adjusted_year_track_ci_95": [
            float(outcome["recommendation_adjusted_year_track_ci_low"]),
            float(outcome["recommendation_adjusted_year_track_ci_high"]),
        ],
        "judge_disagreement_mean_within_startup_sd": float(disagree["mean_within_startup_recommendation_sd"]),
        "trend_slope_2022_2024": float(trend["trend_slope_2022_2024"]),
        "change_2022_to_2024": float(trend["change_2022_to_2024"]),
        "trend_slope_2021_2024_partial": float(trend["trend_slope_2021_2024_partial"]),
        "change_2021_to_2024_partial": float(trend["change_2021_to_2024_partial"]),
        "track_distribution": distribution(group, "Track"),
        "track_overrepresentation": [
            {
                "track": str(row["Track"]),
                "n_applications": int(row["n_applications"]),
                "overrepresentation_ratio": float(row["overrepresentation_ratio"]),
            }
            for _, row in track_overrepresentation.loc[
                track_overrepresentation["concept_id"].eq(concept_id)
            ].sort_values("overrepresentation_ratio", ascending=False).iterrows()
        ],
        "school_distribution": distribution(founder.loc[founder["concept_id"].eq(concept_id)], "harvard_school"),
        "gender_distribution": distribution(founder.loc[founder["concept_id"].eq(concept_id)], "gender"),
        "geography_distribution": distribution(group, "venture location country"),
        "judging_dimension_averages": dimensions,
        "support_pattern": support_row["support_pattern"],
        "support_gaps": {
            "problem_minus_business": float(support_row["problem_minus_business"]),
            "problem_minus_prototype": float(support_row["problem_minus_prototype"]),
            "impact_minus_business": float(support_row["impact_minus_business"]),
        },
        "representative_startups": [{k: example[k] for k in ("venture", "year", "track", "problem_text")} for example in examples],
        "related_concepts": [
            {
                "concept_id": int(row["related_concept_id"]),
                "label": label_map[int(row["related_concept_id"])],
                "jaccard": float(row["jaccard"]),
            }
            for _, row in related.iterrows()
        ],
    }


def _write_profile_markdown(profile: dict[str, object], path: Path) -> None:
    lines = [
        f"# Problem Profile: {profile['problem_label']}", "",
        str(profile["description"]), "",
        f"- Applications: **{profile['n_applications']}** ({profile['share_of_portfolio']:.1%} of eligible portfolio)",
        f"- Raw Recommendation: **{profile['recommendation_mean']:.2f}** (application-level bootstrap 95% CI {profile['recommendation_ci_95'][0]:.2f}–{profile['recommendation_ci_95'][1]:.2f})",
        f"- Year×track-adjusted Recommendation: **{profile['recommendation_adjusted_year_track_mean']:+.2f}** (95% CI {profile['recommendation_adjusted_year_track_ci_95'][0]:+.2f}–{profile['recommendation_adjusted_year_track_ci_95'][1]:+.2f})",
        f"- Judge disagreement: **{profile['judge_disagreement_mean_within_startup_sd']:.2f}** mean within-application SD",
        f"- 2022–2024 annual share trend: **{profile['trend_slope_2022_2024']:+.2%}** per year",
        f"- 2022–2024 share change: **{profile['change_2022_to_2024']:+.2%}**",
        f"- 2021–2024 partial-coverage trend: **{profile['trend_slope_2021_2024_partial']:+.2%}** per year",
        f"- Concept quality: **{profile['quality_badge']}**",
        f"- Support diagnostic: **{profile['support_pattern']}**", "",
        "## Judging dimensions", "",
    ]
    for name, value in profile["judging_dimension_averages"].items():
        lines.append(f"- {name.replace('_', ' ').title()}: **{value:.2f}**")
    lines.extend(["", "## Representative problem statements", ""])
    for example in profile["representative_startups"][:5]:
        lines.extend([f"### {example['venture']} ({example['year']}, {example['track']})", "", str(example["problem_text"]), ""])
    lines.extend(["## Interpretation note", "", str(profile["quality_notes"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(m_concepts: int) -> dict[str, pd.DataFrame | dict[str, object]]:
    startup, judge, assignments, diagnostics = load_analysis_inputs(m_concepts)
    membership, primary = build_membership_long(startup, assignments, diagnostics, m_concepts)
    membership.to_csv(PATHS.tables / f"concept_memberships_m{m_concepts}.csv", index=False)
    primary.to_csv(PATHS.tables / f"primary_problem_areas_m{m_concepts}.csv", index=False)
    outcomes = pd.concat(
        [
            summarize_concept_outcomes(membership, "active_feature", m_concepts),
            summarize_concept_outcomes(primary, "primary_assignment", m_concepts),
        ],
        ignore_index=True,
    )
    trends = build_trends(primary, m_concepts)
    disagreement = analyze_disagreement(primary, m_concepts)
    judge_type = analyze_judge_type(primary, judge, m_concepts)
    founder = build_founder_long(startup, primary, m_concepts)
    composition = composition_tables(primary, founder, m_concepts)
    track_overrepresentation = build_track_overrepresentation(primary, m_concepts)
    lead_gender = lead_gender_composition(primary, m_concepts)
    team_gender = team_gender_composition(founder, m_concepts)
    school_representation = school_representation_index(founder, m_concepts)
    intervention = intervention_profiles(primary, m_concepts)
    strategic = strategic_map(primary, trends, disagreement, m_concepts)

    tables = {
        "concept_outcomes": outcomes,
        "trends": trends,
        "judge_disagreement": disagreement,
        "judge_type": judge_type,
        "founder_demographics_long": founder,
        "intervention_profiles": intervention,
        "strategic_map": strategic,
        "track_overrepresentation": track_overrepresentation,
        "lead_gender_composition": lead_gender,
        "team_gender_composition": team_gender,
        "school_representation": school_representation,
    }
    for name, table in tables.items():
        table.to_csv(PATHS.tables / f"{name}_m{m_concepts}.csv", index=False)
    for name, table in composition.items():
        table.to_csv(PATHS.tables / f"composition_{name.replace(' ', '_')}_m{m_concepts}.csv", index=False)

    if m_concepts == 16:
        write_gender_problem_evaluation_note(primary)

    overlap = pd.read_csv(PATHS.tables / f"concept_overlap_m{m_concepts}.csv")
    stability = pd.read_csv(PATHS.tables / f"concept_stability_m{m_concepts}.csv")
    review = diagnostics[["concept_id", "problem_quality_notes"]].copy()
    review["leadership_ready_example"] = ~review["problem_quality_notes"].str.contains(
        r"geography-driven|heterogeneous|mixed|conflates|not leadership-ready|use cautiously|solution/technology",
        case=False,
        na=False,
    )
    eligible = strategic.loc[~strategic["small_n_flag"]].merge(review, on="concept_id", how="left")
    eligible = eligible.loc[eligible["leadership_ready_example"].fillna(False)]
    chosen = int(eligible.sort_values(["recommendation_mean", "n_startups"], ascending=False).iloc[0]["concept_id"])
    profile = problem_profile(
        chosen,
        primary,
        diagnostics,
        trends,
        disagreement,
        intervention,
        founder,
        overlap,
        stability,
        track_overrepresentation,
        m_concepts,
    )
    profile_path = PATHS.profiles / f"problem_profile_m{m_concepts}_concept_{chosen:02d}.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_profile_markdown(profile, PATHS.profiles / f"problem_profile_m{m_concepts}_concept_{chosen:02d}.md")
    return {**tables, "primary": primary, "membership": membership, "profile": profile}


def write_friday_findings(m_concepts: int = 16) -> Path:
    diagnostics = pd.read_csv(PATHS.tables / f"concept_diagnostics_m{m_concepts}.csv")
    strategic = pd.read_csv(PATHS.tables / f"strategic_map_m{m_concepts}.csv")
    trends = pd.read_csv(PATHS.tables / f"trends_m{m_concepts}.csv")
    disagreement = pd.read_csv(PATHS.tables / f"judge_disagreement_m{m_concepts}.csv")
    intervention = pd.read_csv(PATHS.tables / f"intervention_profiles_m{m_concepts}.csv")
    stability = pd.read_csv(PATHS.tables / f"concept_stability_m{m_concepts}.csv")
    quality = diagnostics[["concept_id", "problem_quality_notes"]].merge(
        stability,
        on="concept_id",
        how="left",
    )
    quality["quality_badge"] = quality.apply(
        lambda row: concept_quality_badge(row, row),
        axis=1,
    )
    leadership_ids = set(
        quality.loc[
            ~quality["small_primary_area"].astype(bool)
            & ~quality["small_and_unstable"].astype(bool)
            & quality["quality_badge"].isin(["Stable", "Stable but broad"]),
            "concept_id",
        ].astype(int)
    )
    eligible = strategic.loc[
        ~strategic["small_n_flag"].astype(bool) & strategic["concept_id"].isin(leadership_ids)
    ].copy()
    largest = eligible.nlargest(1, "n_startups").iloc[0]
    trend_eligible = trends.loc[
        trends["track_scope"].eq("All tracks")
        & ~trends["small_n_flag"].astype(bool)
        & trends["concept_id"].isin(leadership_ids)
    ].merge(eligible[["concept_id", "n_startups"]], on="concept_id", how="left")
    fastest = trend_eligible.nlargest(1, "trend_slope_2022_2024").iloc[0]
    adjusted_median = eligible["recommendation_adjusted_year_track_mean"].median()
    prevalence_median = eligible["prevalence"].median()
    lower_attention = eligible.loc[
        eligible["recommendation_adjusted_year_track_mean"].gt(adjusted_median)
        & eligible["prevalence"].lt(prevalence_median)
    ]
    stronger = lower_attention.nlargest(1, "recommendation_adjusted_year_track_mean").iloc[0]
    support = intervention.loc[
        ~intervention["small_n_flag"].astype(bool) & intervention["concept_id"].isin(leadership_ids)
    ].nlargest(1, "problem_minus_business").iloc[0]
    polarizing = disagreement.loc[
        disagreement["track_scope"].eq("All tracks")
        & ~disagreement["small_n_flag"].astype(bool)
        & disagreement["concept_id"].isin(leadership_ids)
    ].nlargest(1, "mean_within_startup_recommendation_sd").iloc[0]
    lines = [
        "# Friday Meeting Findings", "",
        "This tool gives Harvard i-lab a portfolio-level view of the customer problems founders choose, where historical applications appear to need support, and where judges disagree. "
        "It is designed to help leadership ask sharper programming and portfolio questions—not to rank markets or infer causal effects.", "",
        "All findings use the M=16 primary-assignment leadership taxonomy and equal application weighting. Growth uses 2022–2024 because 2021 has partial structured problem-text coverage (62 of 112 applications); 2021 remains available as historical context but is not treated as fully comparable.", "",
        (
            f"1. **The portfolio is concentrated in a small number of recurring customer problems.**  "
            f"**Evidence:** “{largest['concept_label']}” is the largest established primary area, with "
            f"**N={int(largest['n_startups'])} applications** ({largest['prevalence']:.1%} of the 459 usable application-year observations).  "
            "**Why it might matter:** This provides a concrete baseline for where founder attention and i-lab exposure are already concentrated.  "
            "**Caution:** Primary assignment simplifies an overlapping concept model; applications can activate multiple features."
        ),
        (
            f"2. **One established problem area shows the clearest 2022–2024 attention increase.**  "
            f"**Evidence:** Among areas with at least 10 applications in both endpoint years, “{fastest['concept_label']}” changed by **{fastest['change_2022_to_2024']:+.1%}** of the annual usable portfolio "
            f"from 2022 to 2024, with a simple slope of {fastest['trend_slope_2022_2024']:+.1%} per year "
            f"({int(fastest['count_2022'])} applications in 2022; {int(fastest['count_2024'])} in 2024; **N={int(fastest['n_startups'])} applications** across all years).  "
            "**Why it might matter:** A sustained descriptive shift can prompt curriculum, mentor, or domain-network conversations.  "
            "**Caution:** Three annual points are not a forecast, and changing applicant composition may contribute."
        ),
        (
            f"3. **A lower-attention area received relatively stronger evaluations after year×track adjustment.**  "
            f"**Evidence:** “{stronger['concept_label']}” has adjusted Recommendation **{stronger['recommendation_adjusted_year_track_mean']:+.2f}** "
            f"(application-level bootstrap 95% CI {stronger['recommendation_adjusted_year_track_ci_low']:+.2f} to {stronger['recommendation_adjusted_year_track_ci_high']:+.2f}), "
            f"raw Recommendation {stronger['recommendation_mean']:.2f}/5, and **N={int(stronger['n_startups'])} applications**.  "
            "**Why it might matter:** It is a useful candidate for qualitative follow-up on why relatively favorable evaluations coexist with lower historical attention.  "
            "**Caution:** This is a descriptive within-year-and-track centering, not a causal effect or proof of statistical difference."
        ),
        (
            f"4. **The largest rubric gap points to a commercialization question.**  "
            f"**Evidence:** “{support['concept_label']}” averages {support['problem_customer_definition_mean_mean']:.2f} on Problem & Customer Definition and "
            f"{support['business_model_mean_mean']:.2f} on Business Model, a gap of **{support['problem_minus_business']:+.2f}** "
            f"across **N={int(support['n_startups'])} applications**.  "
            "**Why it might matter:** The pattern can inform questions for programming, mentoring, or curriculum around commercialization.  "
            "**Caution:** Historical judging profiles do not establish that a specific intervention will improve outcomes."
        ),
        (
            f"5. **Judge disagreement is highest in one established problem area.**  "
            f"**Evidence:** “{polarizing['concept_label']}” has mean within-application Recommendation SD **{polarizing['mean_within_startup_recommendation_sd']:.2f}**, "
            f"based on **N={int(polarizing['n_startups'])} applications** and {int(polarizing['n_underlying_ratings'])} underlying ratings.  "
            "**Why it might matter:** High disagreement can identify spaces where evaluation criteria, domain expertise, or risk perspectives merit discussion.  "
            "**Caution:** Disagreement is not synonymous with quality or controversy, and judge mix may differ by year and track."
        ),
    ]
    lines.extend([
        "", "## Method cautions", "",
        "- Recommendation intervals use a fixed-seed, 2,000-resample percentile bootstrap of application-level means.",
        "- The adjusted score subtracts the equally weighted mean application Recommendation within the same year and track; it does not adjust for all selection or judge-composition differences.",
        "- M=32 remains a detailed/exploratory view. Small, unstable, and explicitly mixed concepts are excluded from these headline selections.",
        "- No missing 2021 problem text is backfilled from solution or product descriptions.",
    ])
    path = PATHS.outputs / "FRIDAY_MEETING_FINDINGS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
