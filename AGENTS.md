# Repository instructions

- Treat all application, judge, founder, team, embedding, activation, and excerpt data as confidential.
- Never commit `data/merged_clean.csv`, row-level derived tables, embeddings, model checkpoints, API caches, or secrets.
- Do not modify `/Users/zubeirnoorani/problem-atlas` or the upstream HypotheSAEs source.
- The research unit is a venture application (`Submission ID`), not a deduplicated company.
- Learn concepts from `problem_text` only. Do not use Recommendation or other judging outcomes to select or train concepts.
- Never substitute `Description`, `pitch`, solution, product, or value-proposition text when structured problem-side fields are absent.
- Treat ratings outside 1–5 as missing and exclude declared conflict-of-interest evaluations from aggregates.
- Default analyses weight each application equally. Judge-level analyses must say explicitly when they use judge rows.
- Display sample sizes and missingness for small groups. Do not make causal claims.
- Use random seeds and write a run manifest for every concept model.
- Run tests before updating dashboard artifacts.

