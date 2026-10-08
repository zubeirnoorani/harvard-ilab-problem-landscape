from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402


CRIMSON = "#A51C30"
INK = "#17232D"
TEAL = "#2A6F6B"
GOLD = "#C89B3C"
PARCHMENT = "#F7F3EA"
SLATE = "#71808C"


st.set_page_config(page_title="Harvard i-lab Problem Landscape", page_icon="◈", layout="wide")
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');
    :root {{ --crimson:{CRIMSON}; --ink:{INK}; --paper:{PARCHMENT}; --teal:{TEAL}; }}
    .stApp {{ background: #FCFBF8; color: var(--ink); font-family: 'Libre Franklin', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Source Serif 4', Georgia, serif; color: var(--ink); letter-spacing: -0.02em; }}
    [data-testid="stSidebar"] {{ background: #F3EFE6; border-right: 1px solid #DED8CC; }}
    [data-testid="stMetric"] {{ background: white; border: 1px solid #E4DED3; border-top: 3px solid var(--crimson); padding: 14px 16px; }}
    .eyebrow {{ color: var(--crimson); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
    .deck {{ max-width: 850px; color: #52606A; font-size: 1.08rem; line-height: 1.55; margin-top: -.4rem; }}
    .method-note {{ padding: 14px 16px; border-left: 4px solid var(--teal); background: #EDF4F1; color: #34434A; }}
    .quality-note {{ padding: 12px 14px; border: 1px solid #E1C989; background: #FFF9E8; color: #5A4A22; }}
    div[data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid #DED8CC; }}
    button[data-baseweb="tab"] {{ font-family: 'Libre Franklin'; font-weight: 600; }}
    .stPlotlyChart {{ border: 1px solid #E8E3D9; background: white; padding: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bundle(m: int) -> dict[str, pd.DataFrame]:
    return {
        "primary": pd.read_csv(PATHS.tables / f"primary_problem_areas_m{m}.csv", low_memory=False),
        "diagnostics": pd.read_csv(PATHS.tables / f"concept_diagnostics_m{m}.csv", low_memory=False),
        "overlap": pd.read_csv(PATHS.tables / f"concept_overlap_m{m}.csv"),
        "intervention": pd.read_csv(PATHS.tables / f"intervention_profiles_m{m}.csv"),
        "founders": pd.read_csv(PATHS.tables / f"founder_demographics_long_m{m}.csv", low_memory=False),
        "judge_type": pd.read_csv(PATHS.tables / f"judge_type_m{m}.csv"),
        "disagreement": pd.read_csv(PATHS.tables / f"judge_disagreement_m{m}.csv"),
        "trends": pd.read_csv(PATHS.tables / f"trends_m{m}.csv"),
        "stability": pd.read_csv(PATHS.tables / f"concept_stability_m{m}.csv"),
    }


def style_figure(fig: go.Figure, title: str, subtitle: str = "", height: int = 560) -> go.Figure:
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>" if subtitle else title, "x": 0.02},
        height=height,
        margin={"l": 45, "r": 25, "t": 90, "b": 45},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Libre Franklin, Arial", "color": INK},
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#E9E4DA", zeroline=False)
    fig.update_yaxes(gridcolor="#E9E4DA", zeroline=False)
    return fig


def scope_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    denominators = data.groupby("year")["application_id"].nunique()
    for (concept_id, label), group in data.groupby(["concept_id", "concept_label"]):
        annual = group.groupby("year")["application_id"].nunique().reindex(denominators.index, fill_value=0)
        shares = annual / denominators
        slope = np.polyfit(shares.index, shares.values, 1)[0] if len(shares) >= 2 else np.nan
        rows.append(
            {
                "concept_id": int(concept_id), "concept_label": label,
                "n_startups": group["application_id"].nunique(),
                "prevalence": group["application_id"].nunique() / data["application_id"].nunique(),
                "recommendation_mean": group["recommendation_mean"].mean(),
                "disagreement": group["recommendation_sd"].mean(), "trend": slope,
            }
        )
    return pd.DataFrame(rows)


def constellation(summary: pd.DataFrame, overlap: pd.DataFrame) -> go.Figure:
    ids = set(summary["concept_id"])
    edges = overlap.loc[overlap["concept_a"].isin(ids) & overlap["concept_b"].isin(ids)]
    threshold = edges["jaccard"].quantile(0.82) if len(edges) else 1
    graph = nx.Graph()
    graph.add_nodes_from(sorted(ids))
    for _, row in edges.loc[edges["jaccard"] > threshold].iterrows():
        graph.add_edge(int(row["concept_a"]), int(row["concept_b"]), weight=float(row["jaccard"]))
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    ex, ey = [], []
    for left, right in graph.edges():
        ex += [positions[left][0], positions[right][0], None]
        ey += [positions[left][1], positions[right][1], None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line={"color": "#DDD7CC", "width": 1}, hoverinfo="skip")
    ordered = summary.set_index("concept_id").loc[list(graph.nodes())]
    node_trace = go.Scatter(
        x=[positions[node][0] for node in graph.nodes()], y=[positions[node][1] for node in graph.nodes()],
        mode="markers+text", text=[label if len(label) < 32 else label[:30] + "…" for label in ordered["concept_label"]],
        textposition="top center",
        customdata=np.column_stack([ordered["concept_label"], ordered["n_startups"], ordered["recommendation_mean"], ordered["trend"]]),
        hovertemplate="<b>%{customdata[0]}</b><br>Ventures: %{customdata[1]}<br>Recommendation: %{customdata[2]:.2f}<br>Trend: %{customdata[3]:+.2%}/yr<extra></extra>",
        marker={"size": 14 + 3 * np.sqrt(ordered["n_startups"]), "color": ordered["recommendation_mean"],
                "colorscale": [[0, "#DCEAE7"], [0.55, TEAL], [1, CRIMSON]], "line": {"color": "white", "width": 2},
                "colorbar": {"title": "Recommendation"}},
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False)
    return style_figure(fig, "Problem constellation", "Links show strongest overlaps in multi-concept membership", 650)


st.markdown('<div class="eyebrow">Harvard Innovation Labs · Research prototype</div>', unsafe_allow_html=True)
st.title("The problems Harvard entrepreneurs choose to solve")
st.markdown(
    '<div class="deck">A demand-side view of President’s Innovation Challenge applications, organized by customer need—not technology, product, or industry. Judging outcomes are added only after the problem landscape is learned.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Landscape controls")
    granularity = st.radio("Concept granularity", [16, 32], horizontal=True, help="16 is the leadership view; 32 is a finer diagnostic view.")
    bundle = load_bundle(granularity)
    all_tracks = sorted(bundle["primary"]["Track"].dropna().unique())
    track = st.selectbox("Track", ["All tracks"] + all_tracks)
    years = st.multiselect("Years", [2021, 2022, 2023, 2024], default=[2021, 2022, 2023, 2024])
    min_size = st.slider("Minimum primary-area size", 1, 25, 5)
    st.caption("Primary areas are used for maps. Full multi-concept activations remain in the analytical tables.")

primary = bundle["primary"].copy()
if track != "All tracks":
    primary = primary.loc[primary["Track"].eq(track)]
primary = primary.loc[primary["year"].isin(years)]
summary = scope_summary(primary) if len(primary) else pd.DataFrame()
if len(summary):
    summary = summary.loc[summary["n_startups"] >= min_size]
    primary = primary.loc[primary["concept_id"].isin(summary["concept_id"])]

if primary.empty:
    st.error("No primary problem areas meet the current filters.")
    st.stop()

concept_options = summary.sort_values("n_startups", ascending=False)["concept_label"].tolist()
with st.sidebar:
    selected_label = st.selectbox("Explore a problem", concept_options)
selected_id = int(summary.loc[summary["concept_label"].eq(selected_label), "concept_id"].iloc[0])

tab_landscape, tab_problem, tab_track, tab_founder, tab_judge, tab_method = st.tabs(
    ["Problem Landscape", "Explore a Problem", "Track Comparison", "Founder Landscape", "Judge Evaluation", "Data / Methodology"]
)

with tab_landscape:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eligible ventures", f"{primary['application_id'].nunique():,}")
    c2.metric("Problem areas shown", f"{summary['concept_id'].nunique()}")
    c3.metric("Mean recommendation", f"{primary['recommendation_mean'].mean():.2f}")
    c4.metric("Mean judge disagreement", f"{primary['recommendation_sd'].mean():.2f}")
    st.plotly_chart(constellation(summary, bundle["overlap"]), width="stretch", config={"displayModeBar": False})
    opportunity = px.scatter(
        summary, x="recommendation_mean", y="trend", size="n_startups", color="disagreement", text="concept_label",
        hover_name="concept_label", size_max=55, color_continuous_scale=["#DCEAE7", TEAL, CRIMSON],
        hover_data={"n_startups": True, "prevalence": ":.1%", "trend": ":+.2%"},
    )
    opportunity.update_traces(textposition="top center", marker={"line": {"color": "white", "width": 1.5}})
    opportunity.add_hline(y=0, line_dash="dot", line_color=SLATE)
    opportunity.add_vline(x=summary["recommendation_mean"].median(), line_dash="dot", line_color=SLATE)
    opportunity.update_xaxes(title="Mean startup Recommendation")
    opportunity.update_yaxes(title="Annual change in portfolio share", tickformat="+.1%")
    st.plotly_chart(style_figure(opportunity, "Strategic opportunity map", "Bubble size = ventures; color = judge disagreement"), width="stretch")
    st.markdown('<div class="method-note">The taxonomy is outcome-independent. Recommendation and disagreement are descriptive overlays, and every point exposes its venture count.</div>', unsafe_allow_html=True)

with tab_problem:
    group = primary.loc[primary["concept_id"].eq(selected_id)]
    diagnostic = bundle["diagnostics"].set_index("concept_id").loc[selected_id]
    stability = bundle["stability"].set_index("concept_id").loc[selected_id]
    st.subheader(selected_label)
    st.write(diagnostic["description"])
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Ventures", f"{group['application_id'].nunique()}")
    p2.metric("Portfolio share", f"{group['application_id'].nunique() / primary['application_id'].nunique():.1%}")
    p3.metric("Recommendation", f"{group['recommendation_mean'].mean():.2f}")
    p4.metric("Within-startup judge SD", f"{group['recommendation_sd'].mean():.2f}")
    left, right = st.columns([1.25, 1])
    with left:
        annual = group.groupby("year")["application_id"].nunique().reindex(years, fill_value=0).reset_index(name="ventures")
        annual_fig = px.line(annual, x="year", y="ventures", markers=True, color_discrete_sequence=[CRIMSON])
        annual_fig.update_xaxes(dtick=1)
        st.plotly_chart(style_figure(annual_fig, "Venture attention over time", height=380), width="stretch")
    with right:
        dimensions = pd.DataFrame(
            {
                "dimension": ["Problem / customer", "Solution / prototype", "Business model", "Impact"],
                "mean": [
                    group["problem_customer_definition_mean"].mean(), group["solution_prototype_mean"].mean(),
                    group["business_model_mean"].mean(), group["impact_mean"].mean(),
                ],
            }
        )
        score_fig = px.bar(dimensions, x="mean", y="dimension", orientation="h", range_x=[1, 5], color_discrete_sequence=[TEAL])
        st.plotly_chart(style_figure(score_fig, "Judging profile", height=380), width="stretch")
    dist_left, dist_right = st.columns(2)
    with dist_left:
        tracks = group["Track"].value_counts().reset_index(name="ventures")
        st.plotly_chart(style_figure(px.bar(tracks, x="Track", y="ventures", color_discrete_sequence=[CRIMSON]), "Track mix", height=350), width="stretch")
    with dist_right:
        schools = bundle["founders"].loc[bundle["founders"]["application_id"].isin(group["application_id"]), "harvard_school"].fillna("Missing").value_counts().head(10).reset_index(name="founders")
        st.plotly_chart(style_figure(px.bar(schools, x="founders", y="harvard_school", orientation="h", color_discrete_sequence=[GOLD]), "Founder school mix", height=350), width="stretch")
    st.markdown(f'<div class="quality-note"><b>Concept review note.</b> {diagnostic["problem_quality_notes"]} Alternate-seed stability: <b>{stability["stability_flag"]}</b> (mean matched activation correlation {stability["mean_matched_activation_correlation"]:.2f}).</div>', unsafe_allow_html=True)
    st.markdown("### Representative problem statements")
    for example in json.loads(diagnostic["top_examples"])[:5]:
        with st.expander(f"{example['venture']} · {example['year']} · {example['track']}"):
            st.write(example["problem_text"])

with tab_track:
    all_primary = bundle["primary"].loc[bundle["primary"]["year"].isin(years)]
    matrix = all_primary.groupby(["concept_label", "Track"])["application_id"].nunique().rename("n").reset_index()
    totals = all_primary.groupby("Track")["application_id"].nunique().rename("total").reset_index()
    matrix = matrix.merge(totals, on="Track")
    matrix["share"] = matrix["n"] / matrix["total"]
    pivot = matrix.pivot(index="concept_label", columns="Track", values="share").fillna(0)
    track_fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale=[[0, "white"], [1, CRIMSON]], text=np.round(pivot.values * 100, 1), texttemplate="%{text}%", colorbar={"title": "share"}))
    st.plotly_chart(style_figure(track_fig, "Track emphasis by problem area", "Column-normalized shares; years follow the global filter", max(600, 34 * len(pivot))), width="stretch")

with tab_founder:
    founders = bundle["founders"].loc[bundle["founders"]["application_id"].isin(primary["application_id"])]
    school = founders.groupby(["harvard_school", "concept_label"]).size().rename("n").reset_index()
    school = school.loc[school["harvard_school"].fillna("Missing").ne("Missing")]
    top_schools = school.groupby("harvard_school")["n"].sum().nlargest(12).index
    school = school.loc[school["harvard_school"].isin(top_schools)]
    school_pivot = school.pivot(index="harvard_school", columns="concept_label", values="n").fillna(0)
    school_fig = go.Figure(go.Heatmap(z=school_pivot.values, x=school_pivot.columns, y=school_pivot.index, colorscale=[[0, "white"], [1, CRIMSON]], colorbar={"title": "founders"}))
    st.plotly_chart(style_figure(school_fig, "School × problem area", "Counts of supplied founder/team records", 560), width="stretch")
    gender = founders.groupby(["concept_label", "gender"]).size().rename("n").reset_index()
    gender["share"] = gender["n"] / gender.groupby("concept_label")["n"].transform("sum")
    gender_fig = px.bar(gender, x="share", y="concept_label", color="gender", orientation="h", color_discrete_sequence=[CRIMSON, TEAL, GOLD, SLATE, "#B9B2A5"])
    gender_fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style_figure(gender_fig, "Gender representation", "Supplied variables only; missing retained", max(560, 35 * gender["concept_label"].nunique())), width="stretch")

with tab_judge:
    disagree = primary.groupby(["concept_id", "concept_label"]).agg(
        recommendation=("recommendation_mean", "mean"), disagreement=("recommendation_sd", "mean"), ventures=("application_id", "nunique")
    ).reset_index()
    dfig = px.scatter(disagree, x="recommendation", y="disagreement", size="ventures", text="concept_label", hover_name="concept_label", size_max=55, color="disagreement", color_continuous_scale=[TEAL, GOLD, CRIMSON])
    dfig.update_traces(textposition="top center")
    st.plotly_chart(style_figure(dfig, "Recommendation × judge disagreement", "Each startup contributes one mean and one within-startup SD"), width="stretch")
    intervention = bundle["intervention"].loc[bundle["intervention"]["concept_id"].isin(summary["concept_id"])]
    zcols = ["problem_customer_definition_z", "solution_prototype_z", "business_model_z", "impact_z"]
    heat = go.Figure(go.Heatmap(z=intervention[zcols].values, x=["Problem", "Prototype", "Business model", "Impact"], y=intervention["concept_label"], colorscale="RdBu", zmid=0, text=np.round(intervention[zcols].values, 2), texttemplate="%{text}"))
    st.plotly_chart(style_figure(heat, "Where should i-lab help?", "Standardized relative strengths and weaknesses", max(600, 35 * len(intervention))), width="stretch")
    jt = bundle["judge_type"].loc[bundle["judge_type"]["concept_id"].eq(selected_id)]
    judge_fig = px.bar(jt, x="judge_type", y="recommendation_mean", error_y="recommendation_sd", color="judge_type", hover_data=["n_ratings", "n_startups"], color_discrete_sequence=[CRIMSON, TEAL, GOLD, SLATE])
    judge_fig.update_yaxes(range=[1, 5])
    st.plotly_chart(style_figure(judge_fig, f"Judge types · {selected_label}", "Unadjusted mean; table includes year/track-centered differences", 430), width="stretch")
    st.dataframe(jt[["judge_type", "recommendation_mean", "recommendation_sd", "n_ratings", "n_startups", "mean_year_track_centered_rating", "small_n_flag"]], width="stretch", hide_index=True)

with tab_method:
    st.subheader("What this prototype does")
    st.markdown(
        """
        1. Treats each `Submission ID` as one competition application and removes conflict-of-interest reviews from score aggregation.
        2. Constructs `problem_text` only from structured customer/stakeholder and problem/need fields. Generic venture descriptions are never used as a fallback.
        3. Embeds text locally with `nomic-ai/modernbert-embed-base` and trains the pinned HypotheSAEs top-K sparse autoencoder with K=4 at M=16 and M=32.
        4. Reviews strongest and moderate activations for every feature and flags mixed, geographic, solution-like, or small-N features.
        5. Adds startup-weighted judging, founder, track, and year overlays only after concept learning.
        """
    )
    a1, a2, a3 = st.columns(3)
    a1.metric("Judge rows", "7,281")
    a2.metric("Applications", "509")
    a3.metric("Usable problem texts", "459")
    st.markdown('<div class="quality-note"><b>Confidentiality.</b> This dashboard reads local derived files that contain confidential application text. Do not deploy or share it publicly without Harvard i-lab approval and an explicit data-governance review.</div>', unsafe_allow_html=True)
    st.markdown("### Known limitations")
    st.markdown(
        """
        - Fifty 2021 applications lack structured problem-side text and remain outside the learned landscape.
        - SAE features overlap and are not mutually exclusive clusters; the primary assignment is a visualization convenience.
        - Concept labels are interpretations of activation patterns, not ground truth categories.
        - Differences in judging scores are descriptive and may reflect year, track, judge composition, or selection processes.
        - Small-N and mixed features should not drive programming decisions without qualitative follow-up.
        """
    )
    validation = bundle["diagnostics"].merge(
        bundle["stability"][["concept_id", "n_primary_assignments", "mean_matched_activation_correlation", "mean_matched_membership_jaccard", "stability_flag", "small_and_unstable"]],
        on="concept_id", how="left",
    )
    st.dataframe(validation[["concept_id", "concept_label", "n_startups", "n_primary_assignments", "prevalence", "stability_flag", "small_and_unstable", "problem_quality_notes", "label_status"]], width="stretch", hide_index=True)
