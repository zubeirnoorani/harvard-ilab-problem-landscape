# Harvard i-lab Problem Landscape

This repository builds a reproducible, demand-side view of Harvard Innovation Labs President's Innovation Challenge applications from 2021–2024. It groups venture applications by the customer problem they address, not by technology, product, solution, or industry, and then overlays judging outcomes and founder composition.

## Data structure

The confidential source export is judge-level: one row is one judge's evaluation of one application. `Submission ID` is the application key. The pipeline aggregates ratings to one row per application before any portfolio-level analysis, preventing applications with more judges from receiving extra weight.

Place the supplied export at `data/merged_clean.csv`. That path is ignored by Git.

## Pipeline

1. Audit all columns, missingness, identifiers, duplicates, track/year coverage, and application-field stability.
2. Aggregate judge ratings to one row per application.
3. Build deterministic demand-side `problem_text` from structured problem and customer/stakeholder fields only.
4. Embed problem text locally and train outcome-independent HypotheSAEs models at M=16 and M=32, K=4.
5. Review every feature's strongest and moderate activators before marking it leadership-ready.
6. Overlay judging outcomes, trends, disagreement, judge type, school, gender, geography, and industry.
7. Generate figures, problem profiles, meeting findings, and the Streamlit prototype.

## Setup

Python 3.11–3.13 follows the upstream HypotheSAEs support range. This repository also runs on Python 3.14 by importing an unmodified, pinned checkout directly (the route used for the current artifacts):

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[sae]'
```

Place the confidential export at `data/merged_clean.csv`, then run the complete pipeline:

```bash
.venv/bin/python -m src.pipeline
.venv/bin/pytest
```

The pipeline clones HypotheSAEs into `external/HypotheSAEs`, checks out commit `706d8979c71befe8e3151f51a8d4dbf87d4d7e41`, and never modifies it. The first run downloads the public ModernBERT embedding model; subsequent runs use local embedding and SAE caches. To deliberately rebuild expensive artifacts:

```bash
.venv/bin/python -m src.pipeline --force-embeddings --force-sae
```

Launch the local dashboard:

```bash
.venv/bin/streamlit run dashboard/app.py
```

The notebooks are thin, inspectable companions to the source modules rather than independent implementations.

## Hosted aggregate prototype

The disclosure-controlled leadership view is deployed at:

<https://harvard-ilab-problem-landscape.vercel.app>

The hosted research atlas includes aggregated and track-specific landscapes, M=16/M=32 and primary/overlapping concept views, year filters, problem profiles, judging dimensions, trends, judge disagreement and type, gender, Harvard school, geography, industry, model validation, and data coverage. It contains aggregate cells only and suppresses cells below 10 observations. It does not publish application text, venture names, application IDs, row-level founder attributes, embeddings, activations, or checkpoints. Rebuild its payload after regenerating the analytical outputs with:

```bash
.venv/bin/python scripts/build_public_payload.py
```

The complete confidential drill-down experience remains the local Streamlit dashboard.

## Cost and optional API use

The reproduced pipeline has no per-call API cost: embeddings run locally, the SAEs train locally, and all 48 feature labels were manually reviewed. The only normal costs are local compute, disk space, and network bandwidth for installing dependencies/public model weights. `.env.example` reserves an optional OpenAI labeling configuration with a $5 ceiling, but no API labeling command is enabled in this prototype and no application text was sent to OpenAI.

## Privacy

Raw data, row-level derivatives, problem excerpts, embeddings, activations, checkpoints, and LLM caches remain local and gitignored. Aggregate Markdown findings should be reviewed for small-cell disclosure before sharing. Do not infer gender from names; use only supplied fields.

The public repository contains reproducible code, empty notebooks, disclosure-controlled aggregate reports, and a hosted payload whose analytical cells satisfy N ≥ 10. Access to the confidential source export is still required to reproduce the private row-level and model artifacts locally.

## Known limitations

- Fifty 2021 applications lack structured problem/customer fields and are not backfilled from solution-heavy descriptions.
- Some official track labels disagree with the form family populated by the applicant; the pipeline logs these cases.
- Returning ventures are separate application-year observations.
- SAE features overlap and are not mutually exclusive clusters. `primary_problem_concept` exists only for views that require one assignment.
- Trends cover four annual observations and are descriptive.
- M=32 contains several mixed or geography-driven features; M=16 is the default leadership view, while both remain available for inspection.

## Key outputs

- `outputs/DATA_AUDIT.md` — identifier, field, coverage, duplicate, and missingness audit.
- `outputs/tables/startup_level.csv` — one row per application with startup-weighted judging aggregates.
- `outputs/tables/problem_text.csv` — deterministic demand-side text and quality flags.
- `outputs/tables/concept_diagnostics_m16.csv` and `concept_diagnostics_m32.csv` — reviewed concepts and examples.
- `outputs/figures/` — interactive and static landscape, trend, evaluation, composition, and intervention views.
- `outputs/profiles/` — reusable example Problem Profiles.
- `outputs/FRIDAY_MEETING_FINDINGS.md` — concise leadership discussion findings.
