# Submission Manifest and Legacy-Draft Guard

Date: 2026-07-11

This manifest records which files belong to the current manuscript package and which files are retained only as historical working drafts. Its purpose is to prevent stale results such as the legacy Iterative RAG F1 0.7894 from being accidentally included in submission materials.

Packaging rule: include only files listed in the active package below, plus target-venue style files if needed. Do not package the whole local workspace or unaudited working directories.

## Include in the active manuscript package

- `manuscript_v0.tex`
- `manuscript_v0.pdf` after a fresh clean compile
- `front_matter_v1.tex`
- `introduction_v1.tex`
- `related_work_draft.tex`
- `methods_section_draft.tex`
- `figure1_method_pipeline_tikz.tex`
- `figure2_hyde_mechanism_leakage_tikz.tex`
- `figure3_evidence_error_taxonomy_tikz.tex`
- `tables_main_compact.tex`
- `table_corpus_statistics.tex`
- `table_dedup_retrieval_sensitivity.tex`
- `table_2wiki_generalization.tex`
- `table_efficiency_profile.tex`
- `table_qualitative_case_study.tex`
- `experiments_section_draft.tex`
- `discussion_limitations_v0.tex`
- `conclusion_v0.tex`
- `related_work_references.bib`

## Include only if submitting supplemental material

- `supplemental_material.tex`
- `supplemental_material.pdf` after a fresh clean compile
- `table_corpus_statistics.tex`
- `table_dedup_retrieval_sensitivity.tex`
- `table_reproducibility_checklist.tex`
- `table_model_invocation_ledger.tex`
- `table_qwen_api_reproducibility.tex`
- `table_configuration_provenance.tex`
- `table_efficiency_profile_full.tex`
- `table_hyde_overlap_diagnostic.tex`
- `table_hyde_overlap_robustness.tex`
- `table_answer_absent_subset.tex`
- `table_paired_reliability.tex`
- `table_retrieval_paired_tests.tex`
- `table_hyde_verifier_guard_ablation.tex`
- `table_2wiki_hyde_query_mechanism.tex`
- `table_evidence_completeness_buckets_supp.tex`
- `table_qualitative_case_study.tex`

## Audit supplements, not manuscript text

- `reproducibility_protocol.md`
- `model_invocation_ledger.md`
- `reviewer_response_bank.md`
- `claim_evidence_audit_2026-07-08.md`
- `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl`
- `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl`
- `local-artifacts\results\ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_retrieval.jsonl`
- `local-artifacts\results\ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl`
- `local-artifacts\single_query_reformulation_overlap_audit_hotpotqa_test500.md`
- `local-artifacts\answer_absent_subset_paired_ci.csv`
- `local-artifacts\answer_absent_subset_paired_ci.md`
- `local-artifacts\retrieval_paired_tests.csv`
- `local-artifacts\retrieval_paired_tests.md`
- `src\evaluation\analyze_corpus_statistics.py`
- `local-artifacts\corpus_statistics_test500.csv`
- `local-artifacts\hyde_verifier_guard_ablation.csv`
- `local-artifacts\hyde_verifier_guard_ablation.md`
- `local-artifacts\artifact_grounded_case_studies.md`
- `local-artifacts\dedup_retrieval_sensitivity.csv`
- `local-artifacts\dedup_retrieval_sensitivity.md`
- `src\evaluation\analyze_dedup_retrieval_sensitivity.py`

## Exclude from submission unless explicitly rewritten

These files are retained as historical working drafts, old previews, or local QA artifacts. They may intentionally contain stale method names, legacy values, old terminology, or non-final figure text.

- `front_matter_v0.tex`
- `introduction_v0.tex`
- `title_abstract_introduction_v1.md`
- `related_work_draft.md`
- `methods_section_draft.md`
- `experiments_section_draft.md`
- `manuscript_v0_notes.md`
- `manuscript_storyline_blueprint_v1.md`
- legacy Chinese table/ablation/figure draft matching `*方法流程图草案.md`
- `主实验表_消融表_方法流程图草案.md`
- `figure1_method_pipeline_draft.mmd`
- `figure_candidates/`
- `tables_supplemental_extra.tex`
- local render previews such as `figure*_preview*.png` and `figure*_check_page*.png`
- standalone compile scaffolds such as `*_standalone.tex`, `*.aux`, `*.log`, `*.out`, and old standalone PDFs
- intermediate check builds such as `manuscript_v0_fig2_check.*`, `manuscript_v0_fig3_check.*`, `manuscript_v0_figures_check.*`, and `manuscript_v0_relatedwork_check.*`
- any manually exported draft that was compiled before the canonical Qwen3.7-Max result ledger was frozen

## Final packaging check

Before packaging the manuscript, scan only the active manuscript files rather than the entire draft directory, because historical notes intentionally preserve legacy values:

```powershell
rg -n "0\.7894|0\.7924|legacy reader|qwen3\.7-max-2026-05-17|same-budget|evidence-like|Selective Answer Verification Agent|selective verification agent|evidence-guided iterative RAG pipeline|dominant factor|leakage boundary" manuscript_v0.tex front_matter_v1.tex introduction_v1.tex related_work_draft.tex methods_section_draft.tex figure1_method_pipeline_tikz.tex figure2_hyde_mechanism_leakage_tikz.tex figure3_evidence_error_taxonomy_tikz.tex tables_main_compact.tex table_2wiki_generalization.tex table_efficiency_profile.tex experiments_section_draft.tex discussion_limitations_v0.tex conclusion_v0.tex
```

The active manuscript files should return no hits for these legacy strings. Audit supplements such as this manifest, `reproducibility_protocol.md`, `model_invocation_ledger.md`, and `reviewer_response_bank.md` may intentionally mention legacy values to explain provenance.

## Active Package Source of Truth

The source of truth for the main paper is `manuscript_v0.tex` and the files it directly inputs. The source of truth for supplemental tables is `supplemental_material.tex` and the files it directly inputs. Markdown section drafts are not source of truth unless explicitly regenerated and reconciled with the `.tex` files.
