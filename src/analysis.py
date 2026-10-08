from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .config import PATHS, RATING_COLUMNS, VALID_YEARS


DIMENSIONS = list(RATING_COLUMNS)


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
        row["n_startups"] = int(group["application_id"].nunique())
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
            valid = shares.notna()
            slope = linregress(np.asarray(VALID_YEARS)[valid], shares[valid]).slope if valid.sum() >= 2 else np.nan
            row = {
                "track_scope": track,
                "concept_id": int(concept_id),
                "concept_label": concept_label,
                "trend_slope_share_per_year": float(slope),
                "change_2021_to_2024": float(shares.loc[2024] - shares.loc[2021])
                if pd.notna(shares.loc[2021]) and pd.notna(shares.loc[2024])
                else np.nan,
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
    rows = []
    for _, row in base.iterrows():
        rows.append(
            {
                "application_id": row["application_id"], "year": row["year"], "Track": row["Track"],
                "concept_id": row["concept_id"], "concept_label": row["concept_label"],
                "member_number": 1, "gender": row.get("Gender", "") or "Missing",
                "harvard_school": row.get("School", "") or "Missing",
                "country": row.get("location country", "") or "Missing",
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
                    "gender": str(row.get(f"m{member} gender", "") or "Missing"),
                    "harvard_school": str(row.get(f"m{member} Harvard", "") or "Missing"),
                    "country": str(row.get(f"m{member} location country", "") or "Missing"),
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
            return "Compelling problem / weaker business model"
        if z_solution > 0.35 and z_problem < -0.35:
            return "Stronger prototype / weaker problem definition"
        if z_impact > 0.35 and z_business < -0.35:
            return "Higher impact / weaker commercialization"
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
    rating_cut = out["recommendation_mean"].median()
    size_cut = out["n_startups"].median()
    growth_cut = 0.005
    disagree_cut = out["mean_within_startup_recommendation_sd"].median()

    def labels(row: pd.Series) -> str:
        values = []
        if row["recommendation_mean"] >= rating_cut and row["trend_slope_share_per_year"] > growth_cut:
            values.append("high-rated + growing")
        if row["recommendation_mean"] >= rating_cut and row["n_startups"] < size_cut:
            values.append("high-rated + underrepresented")
        if row["n_startups"] >= size_cut:
            values.append("highly populated")
        if row["trend_slope_share_per_year"] > growth_cut:
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
    m_concepts: int,
) -> dict[str, object]:
    group = primary.loc[primary["concept_id"].eq(concept_id)].copy()
    diagnostic = diagnostics.set_index("concept_id").loc[concept_id]
    trend = trends.loc[(trends["track_scope"].eq("All tracks")) & trends["concept_id"].eq(concept_id)].iloc[0]
    disagree = disagreement.loc[(disagreement["track_scope"].eq("All tracks")) & disagreement["concept_id"].eq(concept_id)].iloc[0]
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
        "n_ventures": int(group["application_id"].nunique()),
        "share_of_portfolio": float(group["application_id"].nunique() / primary["application_id"].nunique()),
        "recommendation_mean": float(group["recommendation_mean"].mean()),
        "recommendation_sd_between_startups": float(group["recommendation_mean"].std(ddof=1)),
        "judge_disagreement_mean_within_startup_sd": float(disagree["mean_within_startup_recommendation_sd"]),
        "trend_share_per_year": float(trend["trend_slope_share_per_year"]),
        "change_2021_to_2024": float(trend["change_2021_to_2024"]),
        "track_distribution": distribution(group, "Track"),
        "school_distribution": distribution(founder.loc[founder["concept_id"].eq(concept_id)], "harvard_school"),
        "gender_distribution": distribution(founder.loc[founder["concept_id"].eq(concept_id)], "gender"),
        "geography_distribution": distribution(group, "venture location country"),
        "judging_dimension_averages": dimensions,
        "support_pattern": intervention.set_index("concept_id").loc[concept_id, "support_pattern"],
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
        f"- Ventures: **{profile['n_ventures']}** ({profile['share_of_portfolio']:.1%} of eligible portfolio)",
        f"- Recommendation: **{profile['recommendation_mean']:.2f}** mean; **{profile['recommendation_sd_between_startups']:.2f}** SD across startups",
        f"- Judge disagreement: **{profile['judge_disagreement_mean_within_startup_sd']:.2f}** mean within-startup SD",
        f"- Annual share trend: **{profile['trend_share_per_year']:+.2%}** per year",
        f"- 2021–2024 share change: **{profile['change_2021_to_2024']:+.2%}**",
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
    }
    for name, table in tables.items():
        table.to_csv(PATHS.tables / f"{name}_m{m_concepts}.csv", index=False)
    for name, table in composition.items():
        table.to_csv(PATHS.tables / f"composition_{name.replace(' ', '_')}_m{m_concepts}.csv")

    overlap = pd.read_csv(PATHS.tables / f"concept_overlap_m{m_concepts}.csv")
    review = diagnostics[["concept_id", "problem_quality_notes"]].copy()
    review["leadership_ready_example"] = ~review["problem_quality_notes"].str.contains(
        r"geography-driven|heterogeneous|mixed|conflates|not leadership-ready|use cautiously|solution/technology",
        case=False,
        na=False,
    )
    eligible = strategic.loc[~strategic["small_n_flag"]].merge(review, on="concept_id", how="left")
    eligible = eligible.loc[eligible["leadership_ready_example"].fillna(False)]
    chosen = int(eligible.sort_values(["recommendation_mean", "n_startups"], ascending=False).iloc[0]["concept_id"])
    profile = problem_profile(chosen, primary, diagnostics, trends, disagreement, intervention, founder, overlap, m_concepts)
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
    stability32 = pd.read_csv(PATHS.tables / "concept_stability_m32.csv")
    missing = pd.read_csv(PATHS.tables / "problem_text_missingness.csv")
    all_trends = trends.loc[trends["track_scope"].eq("All tracks") & ~trends["small_n_flag"]]
    all_disagreement = disagreement.loc[disagreement["track_scope"].eq("All tracks") & ~disagreement["small_n_flag"]]
    top_size = strategic.nlargest(2, "n_startups")
    top_rating = strategic.loc[~strategic["small_n_flag"]].nlargest(2, "recommendation_mean")
    fastest = all_trends.nlargest(2, "trend_slope_share_per_year")
    polarizing = all_disagreement.nlargest(2, "mean_within_startup_recommendation_sd")
    support = intervention.loc[~intervention["small_n_flag"]].copy()
    support = support.loc[support["support_pattern"].ne("Mixed profile")].head(2)
    missing_total = int(missing.loc[missing["problem_text_quality"].eq("low"), "n_applications"].sum())
    lines = [
        "# Friday Meeting Findings", "",
        "These findings are descriptive, use startup-level weighting, and come from the 16-feature leadership view. "
        "Concepts were learned from demand-side problem text without using judge scores. Small-N results are excluded from rankings below.", "",
        f"1. **Structured problem coverage is high after 2021, but incomplete overall.** {459:,} of 509 applications have usable structured problem/customer text; {missing_total} are flagged rather than backfilled from product descriptions.",
    ]
    number = 2
    for _, row in top_size.iterrows():
        lines.append(f"{number}. **Portfolio concentration:** “{row['concept_label']}” is a large primary problem area with {int(row['n_startups'])} ventures ({row['prevalence']:.1%} of eligible applications).")
        number += 1
    for _, row in fastest.iterrows():
        lines.append(f"{number}. **Growing attention:** “{row['concept_label']}” increased by {row['trend_slope_share_per_year']:+.1%} of the annual eligible portfolio per year on a simple linear trend (counts: {int(row['count_2021'])} in 2021; {int(row['count_2024'])} in 2024).")
        number += 1
    for _, row in top_rating.iterrows():
        lines.append(f"{number}. **Stronger evaluations:** “{row['concept_label']}” has a mean startup Recommendation of {row['recommendation_mean']:.2f} across {int(row['n_startups'])} ventures; this is descriptive, not evidence of causal advantage.")
        number += 1
    for _, row in polarizing.iterrows():
        lines.append(f"{number}. **Judge polarization:** “{row['concept_label']}” has mean within-startup Recommendation SD of {row['mean_within_startup_recommendation_sd']:.2f} across {int(row['n_startups'])} ventures ({int(row['n_underlying_ratings'])} ratings).")
        number += 1
    for _, row in support.iterrows():
        lines.append(f"{number}. **Programming signal:** “{row['concept_label']}” is classified as *{row['support_pattern'].lower()}* based on its relative four-dimension profile; this can guide support conversations, not causal claims.")
        number += 1
    lines.extend([
        "", "## Method cautions", "",
        "- A startup may activate several SAE features; the counts above use the highest-activation feature only for legibility.",
        "- The 32-feature view is useful for drill-down but contains smaller and more mixed areas; the 16-feature view is the default leadership summary.",
        f"- Alternate-seed checks flag {int(stability['small_and_unstable'].sum())} of 16 and {int(stability32['small_and_unstable'].sum())} of 32 concepts as both small primary areas and low/moderate stability.",
        "- Geographic and demographic missingness is shown in the dashboard and must remain in denominators.",
        "- Labels are human-reviewed descriptions of strongest and moderate activations; mixed features are flagged in the diagnostic tables.",
    ])
    path = PATHS.outputs / "FRIDAY_MEETING_FINDINGS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
