# Data Audit

## Unit and coverage

- Judge-level rows: **7,281**
- Source columns: **263**
- Unique applications (`Submission ID`): **509**
- Unique reviewer emails: **596**
- Unique populated judge contact IDs: **540**
- Years: **2021, 2022, 2023, 2024**
- Tracks: **Health & Life Science, Open, Social Impact**
- Ratings per application: min **7**, median **15**, mean **14.3**, max **27**

## Validated identifiers

- `Submission ID` is complete and stable within year/track; it is the application key.
- `Contact ID - 18 Digit` is the preferred judge key; normalized reviewer email is the fallback.
- `Account Name` is the complete venture-name field; `Venture Name` is retained as an application-supplied alternate.

## Ratings

- Valid values are 1–5. Zero and blank values are treated as missing.
- Rows declaring `I have a conflict of interest:` = 1: **39**; retained for audit but excluded from aggregates.
- Two later duplicate judge identities are reconciled by keeping the most recent timestamp, leaving **7,240** evaluation rows eligible for aggregation.
- Application-level outcomes use equal application weighting in downstream summaries.

## Data-quality findings

- Repeated application-reviewer groups: **0**
- Repeated application–preferred-judge groups: **2**; the latest timestamp is retained for aggregation.
- Exact-ish duplicate rows: **0**
- Application fields with within-ID variation: **0**
- Applications with structured demand-side text: **459**; missing: **50** (all missing cases are flagged, never backfilled from descriptions).
- Official track/form-family mismatches: **11**; text construction follows the populated form and logs the mismatch.
- Structured problem/customer field coverage varies by form family and year; see `application_field_coverage.csv`.
- Field availability by year and track is explicitly tabulated in `application_field_coverage.csv`; the structured application form changes materially after 2021.
- The source contains direct identifiers and free text. All row-level outputs remain local and gitignored.

## Structured problem-text coverage and longitudinal comparability

2021 has partial structured problem-text coverage (62 of 112 applications) and is preserved as historical context, not treated as fully comparable with 2022–2024. Default growth and decline metrics therefore use 2022–2024; four-year metrics are explicitly labeled partial coverage.

|   year | Track                 |   applications_total |   usable_problem_text |   coverage_share |
|-------:|:----------------------|---------------------:|----------------------:|-----------------:|
|   2021 | Health & Life Science |                   20 |                    14 |          0.7     |
|   2021 | Open                  |                   42 |                    20 |          0.47619 |
|   2021 | Social Impact         |                   50 |                    28 |          0.56    |
|   2022 | Health & Life Science |                   25 |                    25 |          1       |
|   2022 | Open                  |                   57 |                    57 |          1       |
|   2022 | Social Impact         |                   56 |                    56 |          1       |
|   2023 | Health & Life Science |                   29 |                    29 |          1       |
|   2023 | Open                  |                   56 |                    56 |          1       |
|   2023 | Social Impact         |                   41 |                    41 |          1       |
|   2024 | Health & Life Science |                   30 |                    30 |          1       |
|   2024 | Open                  |                   52 |                    52 |          1       |
|   2024 | Social Impact         |                   51 |                    51 |          1       |

## Year and track counts

|   year | Track                 |   judge_rows |   applications |
|-------:|:----------------------|-------------:|---------------:|
|   2021 | Health & Life Science |          299 |             20 |
|   2021 | Open                  |          722 |             42 |
|   2021 | Social Impact         |          475 |             50 |
|   2022 | Health & Life Science |          398 |             25 |
|   2022 | Open                  |          744 |             57 |
|   2022 | Social Impact         |          512 |             56 |
|   2023 | Health & Life Science |          570 |             29 |
|   2023 | Open                  |         1019 |             56 |
|   2023 | Social Impact         |          422 |             41 |
|   2024 | Health & Life Science |          569 |             30 |
|   2024 | Open                  |          939 |             52 |
|   2024 | Social Impact         |          612 |             51 |

## Key application-field missingness

| column_name              |   applications_missing |   percent_missing |
|:-------------------------|-----------------------:|------------------:|
| Account Name             |                      0 |              0    |
| Venture Name             |                     50 |              9.82 |
| School                   |                      0 |              0    |
| Gender                   |                      6 |              1.18 |
| Industries               |                      0 |              0    |
| industry primary         |                     50 |              9.82 |
| venture location country |                     50 |              9.82 |
| problem HLS              |                    408 |             80.16 |
| stakeholders             |                    408 |             80.16 |
| Problem                  |                    151 |             29.67 |
| Customer                 |                    151 |             29.67 |

## Key field mapping

| Purpose | Columns |
|---|---|
| Application ID | `Submission ID` |
| Judge ID | `Contact ID - 18 Digit`; fallback `Reviewer Email Address` |
| Outcomes | `Recommendation`, `Problem &amp; Customer Definition`, `Prototype/MVP`, `Business Model`, `Impact` |
| HLS demand-side text | `problem HLS`, `stakeholders` |
| Open/Social demand-side text | `Problem`, `Customer` |
| Lead demographics | `School`, `Gender`, `degree` |
| Team composition | `m2`–`m5 Harvard`, `m2`–`m5 gender`, team location fields |
| Venture geography | `venture location country`, `venture location state`, `venture location city (us)` |
| Industry | `Industries`, `industry primary`, `industry secondary` |
| Judge type | `judge_type`, raw `Judge Type` |

Generated reproducibly by `src.load_data.create_audit_outputs`.