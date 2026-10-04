# JIIS follow-up reproduction guide

Version: `0.1.0-local-20261004`. Prepared locally on 2026-10-04; no new permanent repository, DOI or public upload exists. The existing review repository does not establish that this local follow-up archive is hosted there.

## Five layers and their limits

| Layer | Inputs / environment | What was checked here / missing inputs |
|---|---|---|
| Manuscript-source compilation | Complete `paper/jiis` source, bibliography, figures, Springer class/style; compatible LaTeX toolchain | Separate from this capsule. Prior source-build receipts in old delivery are historical evidence; this task did not rerun manuscript compilation. |
| Summary-table reconstruction | Included numeric CSV summaries, existing builders; Python 3.10+ standard library | Actually rebuilt legacy main-table snapshot and two open-generator tables in a fresh extracted directory; open tables compared byte-for-byte to frozen tables. Current JIIS CSVs are also supplied for audit, but their private-source builders are not claimed portable. |
| Paired statistical recomputation | Metric-only per-example open-generator exports for both datasets; existing inference helpers | Actually reran all eight contrasts, each n=500, 10,000 paired bootstrap draws with seed13+contrast index, exact McNemar and Holm family8. Every output statistic matched the frozen CSV. This does not cover the 2026-10-03 news/reader families. |
| Raw-prediction rescoring | Original predicted answers, gold answers, exact official and local adapters, null handling | Not executable in lightweight capsule. Historical hosted raw answers are missing; summaries cannot reconstruct them. Private open-reader predictions and upstream gold texts are excluded. |
| Model rerun | Authorized corpora, frozen sample IDs, weights/tokenizers and revision, index, prompts/budgets/adapters, runtime and hardware; hosted service access where used | Not executed here. No GPU, Elasticsearch, hosted inference or new reader call used. Table/metric success must not be described as full experimental reproduction. |

## Minimal offline capsule commands

Extract `release/jiis_reproduction_followup_local_20261004/jiis_reproduction_followup_0.1.0-local-20261004.zip` into a new directory, then run from its `capsule` directory:

```powershell
python -S run.py
```

`-S` disables third-party site-packages to verify standard-library independence. Expected exit code: 0. Expected final message: `PASS: eight paired contrasts match; two open-generator tables byte-identical; legacy table rebuilt.` Outputs: `outputs/legacy_main_tables.tex`, `outputs/table_open_generator_query2doc.tex`, `outputs/table_open_generator_leakage.tex`, `outputs/open_generator_paired_recomputed.csv`, `outputs/verification.json`. The wrapper validates all manifest SHA-256 values before analysis. Do not edit input files to make verification pass.

The existing table builders can also be invoked separately inside the capsule:

```powershell
python -S scripts/rebuild_main_tables_from_summaries.py --root . --out outputs/legacy_main_tables.tex
python -S scripts/build_expert_systems_latex_tables.py --out_main outputs/table_open_generator_query2doc.tex --out_leakage outputs/table_open_generator_leakage.tex
```

## Result/input map

Every supplied file and source hash is enumerated in `manifest.json`; open-generator source-to-export hashes and retained fields are in `export_provenance.json`. The tables below describe applicability rather than claiming every current paper table was independently recomputed.

| Result family | Included input | Expected output / reproduction gap |
|---|---|---|
| Historical HotpotQA main, query composition, equal-budget and supporting-paragraph summaries | `artifacts/summaries/ircot_test500_qwen_results.csv`, `hyde_mechanism_ablation_hotpotqa_test500.csv`, `equal_budget_query_diagnostic/*.csv`, `supporting_paragraph_audit/supporting_paragraph_metrics.csv` | Legacy snapshot via first builder; historical frozen aggregates only, no raw hosted-answer rescoring. |
| Open-generator Dense/HyDE/Query2doc, HotpotQA and 2Wiki | `metrics/hotpotqa.jsonl`, `metrics/2wiki.jsonl`, `artifacts/summaries/expert_systems_minimum/{retrieval_summary,paired_contrasts,leakage_summary}.csv` | Eight paired retrieval contrasts genuinely recomputed; two tables rebuilt. Leakage strata are read from summaries, not re-extracted from generated text. |
| Latest post hoc JIIS news/reader families | `artifacts/summaries/jiis_priority_revision_20261003/paired/{paired_comparisons,prompt_subset_descriptive}.csv` | Frozen per-family means, delta, CI, discordance and p values auditable; exact per-question reader scores, effective inputs, private predictions and source texts missing from capsule, so full reanalysis unavailable. |
| Reader no-retrieval/gold controls | `.../reader_controls/{descriptive_summary,paired_comparisons}.csv` | Frozen summary audit only; n=100 is old500 subset, not independent validation. Raw gold-control text and predictions excluded. |
| Independent64 selected-method check | `.../independent_validation/{independent_results,descriptive_metrics,stage_costs,support_per_example_sanitized}.csv` | Sanitized per-example scores/support diagnostics inspectable; selected policies identical, so no additional policy-advantage test is warranted. Full candidate matrix and text/predictions absent. |
| Token interface and canonical-null/scoring diagnoses, other full-wiki and news results | Existing project summaries/provenance outside capsule, including `artifacts/summaries/jiis_priority_revision_20261003` and `jiis_acceptance_v3` | Not all result artifacts included; see full repository/current supplement source hashes. Actual-token verification requires exact tokenizer/input token IDs and source support spans; official answer rescoring requires excluded predictions/gold. Do not infer these from aggregate tables. |

For complete-model runners refer to the repository's `experiments/run_expert_systems_minimum_enhancement.ps1`, `experiments/run_ircot_fullwiki_formal.ps1`, `src/ircot_fullwiki/run_fullwiki_reader.py`, and `src/multihop_rag/`. Their dependencies and local inputs must be acquired separately; they are not offline capsule commands.

## Public file boundary

`manifest.json` is an explicit allowlist, not a directory-wide permission. Included: project-authored builders/helper code; software license; numerical aggregate CSVs; sanitized metric-only exports; expected open-generator tables; this guide and notices. Excluded: corpora/questions/gold answers, text-containing news retrieval JSONL, generated text, predictions, effective chat text, model weights, private endpoints/credentials, caches and all old release archives. In particular, local `artifacts/per_example/jiis_acceptance_v3/mhrag_*_top10.jsonl` contains article text and must not be swept into a public archive.

See `THIRD_PARTY_NOTICES.md` for official license sources, acquisition routes and unresolved news-corpus rights. The capsule's metric exports remove retrieval metadata and titles and retain only schema/dataset/example IDs/method/numeric metrics. A local archive is not permission to redistribute unreviewed third-party text.

## Verification evidence and archive identity

The clean-run receipt records Python identity, commands, process exit, extracted input hashes and output hashes. `archive_inventory.json` records the archive SHA-256 and inventory. It is a local preservation artifact, with no DOI. Author-controlled permanent archival deposition remains outstanding; update the manuscript only after a real persistent record exists.
