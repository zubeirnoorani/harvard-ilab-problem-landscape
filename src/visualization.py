from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import PATHS, VALID_YEARS


CRIMSON = "#A51C30"
INK = "#17232D"
TEAL = "#2A6F6B"
GOLD = "#C89B3C"
PARCHMENT = "#F7F3EA"
SLATE = "#71808C"


def apply_plotly_style(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    full_title = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        title={"text": full_title, "x": 0.02, "xanchor": "left"},
        font={"family": "Avenir Next, Avenir, Helvetica, Arial, sans-serif", "color": INK},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin={"l": 60, "r": 30, "t": 95, "b": 60},
        legend_title_text="",
        hoverlabel={"font_family": "Avenir Next, Avenir, Helvetica, Arial, sans-serif"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E7E2D8", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E7E2D8", zeroline=False)
    return fig


def _save(fig: go.Figure, stem: str) -> None:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    fig.write_html(PATHS.figures / f"{stem}.html", include_plotlyjs="cdn", full_html=True)


def _strategic_for_scope(primary: pd.DataFrame, track_scope: str) -> pd.DataFrame:
    scope = primary if track_scope == "All tracks" else primary.loc[primary["Track"].eq(track_scope)]
    rows = []
    for (concept_id, concept_label), group in scope.groupby(["concept_id", "concept_label"]):
        counts = group.groupby("year")["application_id"].nunique().reindex(VALID_YEARS, fill_value=0)
        denominators = scope.groupby("year")["application_id"].nunique().reindex(VALID_YEARS, fill_value=0)
        shares = counts / denominators.replace(0, np.nan)
        valid = shares.notna()
        slope = np.polyfit(np.asarray(VALID_YEARS)[valid], shares[valid], 1)[0] if valid.sum() >= 2 else np.nan
        rows.append(
            {
                "concept_id": int(concept_id), "concept_label": concept_label,
                "n_startups": int(group["application_id"].nunique()),
                "prevalence": group["application_id"].nunique() / scope["application_id"].nunique(),
                "recommendation_mean": group["recommendation_mean"].mean(),
                "judge_disagreement": group["recommendation_sd"].mean(),
                "trend_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def opportunity_map(primary: pd.DataFrame, m_concepts: int, track_scope: str = "All tracks") -> go.Figure:
    data = _strategic_for_scope(primary, track_scope)
    fig = px.scatter(
        data,
        x="recommendation_mean",
        y="trend_slope",
        size="n_startups",
        color="judge_disagreement",
        text="concept_label",
        hover_name="concept_label",
        hover_data={"n_startups": True, "prevalence": ":.1%", "recommendation_mean": ":.2f", "trend_slope": ":+.2%", "judge_disagreement": ":.2f"},
        color_continuous_scale=["#DCEAE7", TEAL, CRIMSON],
        size_max=55,
    )
    fig.update_traces(textposition="top center", textfont_size=10, marker={"line": {"color": "white", "width": 1.5}})
    fig.add_hline(y=0, line_dash="dot", line_color=SLATE)
    fig.add_vline(x=data["recommendation_mean"].median(), line_dash="dot", line_color=SLATE)
    fig.update_xaxes(title="Mean startup Recommendation (1–5)")
    fig.update_yaxes(title="Annual change in portfolio share")
    apply_plotly_style(
        fig,
        f"Problem opportunity map · {track_scope}",
        f"M={m_concepts}; bubble size = primary-area ventures; color = judge disagreement. Descriptive, not causal.",
    )
    _save(fig, f"opportunity_map_m{m_concepts}_{track_scope.lower().replace(' ', '_').replace('&', 'and')}")
    return fig


def prevalence_rating_map(primary: pd.DataFrame, m_concepts: int) -> go.Figure:
    data = _strategic_for_scope(primary, "All tracks")
    fig = px.scatter(
        data, x="prevalence", y="recommendation_mean", size="n_startups", color="trend_slope",
        text="concept_label", hover_name="concept_label", color_continuous_scale="RdBu",
        color_continuous_midpoint=0, size_max=55,
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_xaxes(title="Share of eligible portfolio", tickformat=".0%")
    fig.update_yaxes(title="Mean startup Recommendation (1–5)")
    apply_plotly_style(fig, "Attention × evaluation", f"M={m_concepts}; color = annual change in portfolio share")
    _save(fig, f"prevalence_rating_m{m_concepts}")
    return fig


def trend_figure(primary: pd.DataFrame, m_concepts: int, top_n: int = 10) -> go.Figure:
    counts = primary.groupby(["year", "concept_label"])["application_id"].nunique().rename("n").reset_index()
    totals = primary.groupby("year")["application_id"].nunique().rename("total").reset_index()
    data = counts.merge(totals, on="year")
    data["share"] = data["n"] / data["total"]
    largest = primary["concept_label"].value_counts().head(top_n).index
    data = data.loc[data["concept_label"].isin(largest)]
    fig = px.line(data, x="year", y="share", color="concept_label", markers=True, hover_data={"n": True, "share": ":.1%"})
    fig.update_xaxes(dtick=1, title="Competition year")
    fig.update_yaxes(title="Share of eligible startups", tickformat=".0%")
    apply_plotly_style(fig, "How entrepreneurial attention changed", f"Ten largest primary problem areas · M={m_concepts}")
    _save(fig, f"year_trends_m{m_concepts}")
    return fig


def judging_heatmap(intervention: pd.DataFrame, m_concepts: int, standardized: bool = False) -> go.Figure:
    if standardized:
        columns = ["problem_customer_definition_z", "solution_prototype_z", "business_model_z", "impact_z"]
        labels = ["Problem / customer", "Solution / prototype", "Business model", "Impact"]
        zmid, colors = 0, "RdBu"
        title = "Where should i-lab help?"
        subtitle = "Standardized problem-area profile; red/blue indicate relative weakness/strength"
    else:
        columns = [
            "problem_customer_definition_mean_mean", "solution_prototype_mean_mean",
            "business_model_mean_mean", "impact_mean_mean",
        ]
        labels = ["Problem / customer", "Solution / prototype", "Business model", "Impact"]
        zmid, colors = None, [[0, "#F4E5E8"], [0.5, "#F7F3EA"], [1, TEAL]]
        title = "Judging scores by problem area"
        subtitle = "Equal startup weighting; raw mean scores on 1–5 scale"
    data = intervention.sort_values("problem_customer_definition_mean_mean")
    matrix = data[columns].to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=labels, y=data["concept_label"], colorscale=colors, zmid=zmid,
            text=np.round(matrix, 2), texttemplate="%{text}", colorbar={"title": "z-score" if standardized else "mean"},
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(620, 34 * len(data)))
    apply_plotly_style(fig, title, f"{subtitle} · M={m_concepts}")
    _save(fig, f"{'support_diagnostic' if standardized else 'judging_scores'}_m{m_concepts}")
    return fig


def disagreement_figure(disagreement: pd.DataFrame, m_concepts: int) -> go.Figure:
    data = disagreement.loc[disagreement["track_scope"].eq("All tracks")]
    fig = px.scatter(
        data, x="recommendation_mean", y="mean_within_startup_recommendation_sd", size="n_startups",
        color="rating_disagreement_quadrant", text="concept_label", hover_name="concept_label", size_max=55,
        color_discrete_map={
            "High rating / Low disagreement": TEAL,
            "High rating / High disagreement": GOLD,
            "Low rating / Low disagreement": SLATE,
            "Low rating / High disagreement": CRIMSON,
        },
    )
    fig.update_traces(textposition="top center", textfont_size=10)
    fig.update_xaxes(title="Mean startup Recommendation")
    fig.update_yaxes(title="Mean within-startup Recommendation SD")
    apply_plotly_style(fig, "Where judges agree—and disagree", f"M={m_concepts}; bubble size = startups")
    _save(fig, f"judge_disagreement_m{m_concepts}")
    return fig


def school_problem_heatmap(school_table: pd.DataFrame, m_concepts: int) -> go.Figure:
    clean = school_table.loc[school_table["harvard_school"].ne("Missing")].copy()
    common_schools = clean.groupby("harvard_school")["n_founders"].sum().nlargest(12).index
    clean = clean.loc[clean["harvard_school"].isin(common_schools)]
    matrix = clean.pivot_table(index="harvard_school", columns="concept_label", values="share_within_problem", fill_value=0)
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, colorscale=[[0, "#FFFFFF"], [1, CRIMSON]],
            colorbar={"title": "share"}, hovertemplate="%{y}<br>%{x}<br>%{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(height=600)
    fig.update_xaxes(tickangle=-35)
    apply_plotly_style(fig, "Harvard school × problem area", f"Founder share within each primary area · M={m_concepts}; missing shown separately in data")
    _save(fig, f"school_problem_matrix_m{m_concepts}")
    return fig


def gender_figure(gender_table: pd.DataFrame, m_concepts: int) -> go.Figure:
    data = gender_table.copy()
    fig = px.bar(
        data, x="share_within_problem", y="concept_label", color="gender", orientation="h",
        hover_data={"n_founders": True, "share_within_problem": ":.1%"},
        color_discrete_sequence=[CRIMSON, TEAL, GOLD, SLATE, "#B9B2A5"],
    )
    fig.update_xaxes(title="Share of observed founder records", tickformat=".0%")
    fig.update_yaxes(title=None)
    fig.update_layout(barmode="stack", height=max(650, 35 * data["concept_label"].nunique()))
    apply_plotly_style(fig, "Gender representation by problem area", f"Supplied gender variables only; missing retained · M={m_concepts}")
    _save(fig, f"gender_problem_m{m_concepts}")
    return fig


def problem_constellation(
    strategic: pd.DataFrame,
    overlap: pd.DataFrame,
    m_concepts: int,
) -> go.Figure:
    graph = nx.Graph()
    for _, row in strategic.iterrows():
        graph.add_node(int(row["concept_id"]))
    positive = overlap.loc[overlap["jaccard"] > overlap["jaccard"].quantile(0.82)]
    for _, row in positive.iterrows():
        graph.add_edge(int(row["concept_a"]), int(row["concept_b"]), weight=float(row["jaccard"]))
    positions = nx.spring_layout(graph, seed=42, weight="weight", k=1.1 / np.sqrt(max(1, m_concepts)))
    edge_x, edge_y = [], []
    for left, right in graph.edges():
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1, "color": "#D8D2C7"}, hoverinfo="skip")
    ordered = strategic.set_index("concept_id").loc[list(graph.nodes())]
    node_trace = go.Scatter(
        x=[positions[node][0] for node in graph.nodes()],
        y=[positions[node][1] for node in graph.nodes()],
        mode="markers+text",
        text=[textwrap.shorten(label, width=30, placeholder="…") for label in ordered["concept_label"]],
        textposition="top center",
        customdata=np.column_stack([
            ordered["concept_label"], ordered["n_startups"], ordered["recommendation_mean"], ordered["trend_slope_share_per_year"]
        ]),
        hovertemplate="<b>%{customdata[0]}</b><br>Ventures: %{customdata[1]}<br>Recommendation: %{customdata[2]:.2f}<br>Trend: %{customdata[3]:+.2%}/yr<extra></extra>",
        marker={
            "size": 14 + 2.2 * np.sqrt(ordered["n_startups"]),
            "color": ordered["recommendation_mean"],
            "colorscale": [[0, "#D9E4E2"], [0.5, TEAL], [1, CRIMSON]],
            "line": {"width": 2, "color": "white"},
            "colorbar": {"title": "Recommendation"},
        },
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=760, showlegend=False)
    apply_plotly_style(fig, "The i-lab problem constellation", f"M={m_concepts}; proximity reflects overlapping SAE membership, not industry similarity")
    _save(fig, f"problem_constellation_m{m_concepts}")
    return fig


def _static_opportunity(primary: pd.DataFrame, m_concepts: int) -> None:
    data = _strategic_for_scope(primary, "All tracks")
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="white")
    scatter = ax.scatter(
        data["recommendation_mean"], data["trend_slope"],
        s=70 + data["n_startups"] * 18, c=data["judge_disagreement"], cmap="RdYlBu_r",
        alpha=0.85, edgecolor="white", linewidth=1.5,
    )
    for _, row in data.iterrows():
        ax.annotate(textwrap.fill(row["concept_label"], 22), (row["recommendation_mean"], row["trend_slope"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, color=SLATE, linestyle=":")
    ax.axvline(data["recommendation_mean"].median(), color=SLATE, linestyle=":")
    ax.set_xlabel("Mean startup Recommendation (1–5)")
    ax.set_ylabel("Annual change in portfolio share")
    ax.set_title(f"Harvard i-lab problem opportunity map · M={m_concepts}", loc="left", color=INK, fontsize=18, pad=20)
    ax.grid(color="#E7E2D8", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(scatter, ax=ax, label="Mean within-startup Recommendation SD")
    fig.tight_layout()
    fig.savefig(PATHS.figures / f"opportunity_map_m{m_concepts}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations(m_concepts: int) -> dict[str, go.Figure]:
    primary = pd.read_csv(PATHS.tables / f"primary_problem_areas_m{m_concepts}.csv", low_memory=False)
    intervention = pd.read_csv(PATHS.tables / f"intervention_profiles_m{m_concepts}.csv")
    disagreement = pd.read_csv(PATHS.tables / f"judge_disagreement_m{m_concepts}.csv")
    school = pd.read_csv(PATHS.tables / f"composition_harvard_school_m{m_concepts}.csv")
    gender = pd.read_csv(PATHS.tables / f"composition_gender_m{m_concepts}.csv")
    strategic = pd.read_csv(PATHS.tables / f"strategic_map_m{m_concepts}.csv")
    overlap = pd.read_csv(PATHS.tables / f"concept_overlap_m{m_concepts}.csv")
    figures = {
        "constellation": problem_constellation(strategic, overlap, m_concepts),
        "opportunity": opportunity_map(primary, m_concepts),
        "prevalence_rating": prevalence_rating_map(primary, m_concepts),
        "trends": trend_figure(primary, m_concepts),
        "judging": judging_heatmap(intervention, m_concepts),
        "support": judging_heatmap(intervention, m_concepts, standardized=True),
        "disagreement": disagreement_figure(disagreement, m_concepts),
        "school": school_problem_heatmap(school, m_concepts),
        "gender": gender_figure(gender, m_concepts),
    }
    for track in sorted(primary["Track"].dropna().unique()):
        figures[f"opportunity_{track}"] = opportunity_map(primary, m_concepts, track)
    _static_opportunity(primary, m_concepts)
    return figures
