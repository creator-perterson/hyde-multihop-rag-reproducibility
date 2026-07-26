# Claim-Evidence Audit and Pre-Packaging Check

Date: 2026-07-08

Scope: active manuscript files included by `manuscript_v0.tex`, supplemental table driver `supplemental_material.tex`, the result ledger, model invocation ledger, reviewer response bank, and final packaging manifest.

2026-07-09 addendum: a query-composition control was added after this audit. The new row retrieves with the original question plus the generated rewritten query. It reaches all-support hit@10 = 0.6280 and EM/F1 = 0.5800/0.7048, while HyDE-style question + hypothetical passage reaches all-support hit@10 = 0.8680 and EM/F1 = 0.6840/0.8105. This strengthens the claim that the HyDE gain is not explained by simply retaining the original question in the dense query.

2026-07-09 addendum 2: an answer-absent paired CI diagnostic was added for HotpotQA and 2Wiki. On examples where the normalized gold-answer string is absent from the generated HyDE passage, HyDE still improves over Dense by +0.0620 F1 [0.0170, 0.1080] on HotpotQA and +0.2019 F1 [0.1372, 0.2671] on 2Wiki. This reduces the sufficiency of exact answer-string overlap as the only explanation, while preserving the explicit answer-string-overlap boundary.

2026-07-09 addendum 3: a HyDE-base verifier guard ablation was added from existing verifier outputs. Raw verifier outputs produce 2 wrong-to-correct and 2 correct-to-wrong transitions; the numeric-unit guard removes the observed correct-to-wrong transitions and recovers the final 2 wrong-to-correct / 0 correct-to-wrong result. This supports the guard as a safety boundary rather than a major performance source.

2026-07-09 addendum 4: the qualitative case-study table was regenerated from frozen per-example artifacts using `extract_case_study_examples.py`. The selected cases now cover HyDE evidence acquisition, verifier guard harm prevention, and a remaining reader limitation with full supporting evidence.

2026-07-09 addendum 5: retrieval-side paired tests were added from frozen top-10 retrieval artifacts. On HotpotQA, HyDE improves all-support hit@10 over Dense by +0.2380 [0.2000, 0.2780] and over the matched-call Single-query baseline by +0.2500 [0.2140, 0.2860]. The question + hypothetical passage control improves over question + rewritten query by +0.2400 [0.2020, 0.2780]. On 2Wiki, HyDE improves all-support hit@10 over Dense by +0.3100 [0.2680, 0.3520] and over Hybrid by +0.2200 [0.1800, 0.2620]. This makes the evidence-acquisition claim match retrieval-side paired statistics, not only answer-side F1 paired tests.

2026-07-12 addendum: main/supplemental table triage was updated. The four-row 2WikiMultihopQA result table is now part of the active main manuscript, while the full nine-row prompt-character efficiency profile is moved to the supplemental material. The main manuscript keeps a compact four-row call-count table so the lightweight claim remains visible without crowding out the secondary dataset evidence.

## Executive status

No blocking claim-evidence mismatch was found in the active manuscript after this audit. The main claims are now bounded and traceable to tables, paired analyses, supplemental diagnostics, or explicit limitations.

Remaining pre-submission items are mostly packaging and venue-format issues:

- `front_matter_v1.tex` still contains `Author Name`, `Affiliation`, and `email@example.com`; replace these for non-anonymous submission or convert them to the target venue's anonymous format.
- The active manuscript compiles, but the log contains normal underfull box warnings from narrow two-column layout. No overfull boxes were found in the final log scan.
- The 2Wiki generalization table is now included as an active main-manuscript table and captioned as a secondary generalization check.
- Real provider-side latency, request IDs, and billing logs are still unavailable. The manuscript correctly limits the efficiency claim to call counts and prompt-character counts.

## Main claim-evidence map

| Claim | Evidence anchor | Audit status |
| --- | --- | --- |
| Multi-hop RAG is limited by incomplete supporting evidence under this processed-data protocol. | Main results table; evidence-completeness buckets in `experiments_section_draft.tex`; supplemental bucket analysis. | Supported, with bounded wording. |
| HyDE-style document-like query expansion improves HotpotQA evidence acquisition and reader F1 over tested baselines. | `tab:main_results`: HyDE all-hit 0.8680, recall 0.9280, F1 0.8105; retrieval paired tests: HyDE over Dense all-hit +0.2380 [0.2000, 0.2780], recall +0.1200 [0.1010, 0.1410]; Dense F1 0.7199; Hybrid F1 0.7577; Iterative F1 0.7565. | Supported for the tested protocol. |
| The gain is not explained by one extra query-side LLM call alone or by merely retaining the original question. | Matched-call Single-query Reformulation RAG: all-hit 0.6180, F1 0.7130; retrieval paired HyDE-over-single-query all-hit +0.2500 [0.2140, 0.2860]; Question + rewritten query: all-hit 0.6280, F1 0.7048; paired HyDE-over-question+rewritten all-hit +0.2400 [0.2020, 0.2780] and F1 +0.1057 [0.0791, 0.1364]. | Supported; matched-call and query-composition controls now visible in Abstract, Introduction, Experiments, Discussion, and tables. |
| HyDE-style generation is a mechanism adaptation, not a newly invented HyDE method. | Related Work and Discussion explicitly state HyDE is prior work and position this paper as a controlled multi-hop adaptation/audit. | Supported and appropriately modest. |
| Hypothetical passages are not reader evidence. | Methods and Reproducibility Protocol state that the hypothetical passage is used only for retrieval query expansion; reader sees retrieved passages only. | Supported by method description and artifact ledger. |
| Query-side answer-string overlap is a central limitation. | HotpotQA overlap: 292/500 answer-string overlaps, 252 non-trivial; 2Wiki overlap: 347/500, 334 non-trivial; answer-absent paired slices: HotpotQA F1 +0.0620 [0.0170, 0.1080], 2Wiki F1 +0.2019 [0.1372, 0.2671]. | Supported and disclosed. Cannot prove absence of parametric memorization, paraphrased answer content, or benchmark contamination; manuscript states this boundary. |
| Evidence completeness is a major bottleneck, not a causally proven universal law. | Pearson r is presented as descriptive; answer-side paired bootstrap, retrieval-side paired tests, McNemar transitions, and evidence buckets provide stronger support. | Supported after claim softening. Avoid the phrase "dominant factor" as a hard causal claim. |
| Conservative verification is a guarded canonicalization layer, not the main performance driver. | Verifier table: HyDE F1 0.8105 to 0.8135; final 2 W->C, 0 C->W; HyDE guard ablation: raw verifier 2 W->C and 2 C->W, numeric-guarded 2 W->C and 0 C->W; diagnostic verifier variants caused harmful transitions. | Supported and correctly bounded. |
| Qualitative cases are illustrative but artifact-grounded. | `extract_case_study_examples.py` selects cases from Dense/HyDE answer files, HyDE answer-string-overlap audit, and verifier guard-ablation details; output table is `table_qualitative_case_study.tex`. | Supported; do not present as systematic human evaluation. |
| Lightweight cost is explicit and bounded. | Main `tab:efficiency_profile`: HyDE 2.00 calls/q, 1000 total; HyDE+verifier 2.24 calls/q, 1120 total; no training. Supplemental full efficiency table reports prompt-character and evidence-character counts. | Supported as call/character-count evidence only. No latency or billing claim should be made. |
| 2Wiki provides a secondary generalization check. | Main `tab:2wiki_generalization`: HyDE F1 0.7289 vs Dense 0.5767, BM25 0.6169, Hybrid 0.6220; answer paired bootstrap in Experiments; retrieval paired tests: HyDE over Dense all-hit +0.3100 [0.2680, 0.3520], HyDE over Hybrid all-hit +0.2200 [0.1800, 0.2620]. | Supported as secondary evidence, not a full benchmark generalization proof. |

## Active-file risk scan

Commands run from `paper/latex`:

```powershell
rg -n "state-of-the-art|SOTA|prove|proves|demonstrate|dominant factor|answer-overlap-free|cost-free|cheap|low-cost|real-time|latency|billing|significant|novel|new method|full-wiki|0\.7894|0\.7924|qwen3\.7-max-2026-05-17|legacy reader" front_matter_v1.tex introduction_v1.tex related_work_draft.tex methods_section_draft.tex experiments_section_draft.tex discussion_limitations_v0.tex conclusion_v0.tex tables_main_compact.tex table_2wiki_generalization.tex table_efficiency_profile.tex
```

Outcome:

- No active manuscript hit for legacy result values `0.7894` or `0.7924`.
- No active manuscript hit for the legacy model snapshot string `qwen3.7-max-2026-05-17`.
- Strong words that remain are used in bounded contexts, e.g., "not as retrieval free of answer-string overlap", "does not claim measured latency or billing", or "descriptive evidence rather than proof of causality".
- Edited `experiments_section_draft.tex` to replace "low-cost robustness slice" with "no-new-LLM robustness slice" so cost wording does not overreach.

## Result/artifact existence check

The following key artifacts were checked and exist:

- `paper\latex\manuscript_v0.tex`
- `paper\latex\manuscript_v0.pdf`
- `paper\latex\supplemental_material.tex`
- `paper\latex\supplemental_material.pdf`
- `paper\latex\related_work_references.bib`
- `paper\latex\reproducibility_protocol.md`
- `paper\latex\model_invocation_ledger.md`
- `paper\latex\reviewer_response_bank.md`
- `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl`
- `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl`
- `local-artifacts\single_query_reformulation_overlap_audit_hotpotqa_test500.md`

Row-count spot check:

- Single-query reformulation generation JSONL: 500 rows.
- Single-query reformulation reader JSONL: 500 rows.
- HyDE reader JSONL: 500 rows.
- HyDE + conservative verifier merged JSONL: 500 rows.

## Build and reference check

Commands run:

```powershell
xelatex -interaction=nonstopmode -halt-on-error manuscript_v0.tex
xelatex -interaction=nonstopmode -halt-on-error manuscript_v0.tex
xelatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
xelatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
```

Build outputs:

- `manuscript_v0.pdf`: generated successfully, 11 pages, 125404 bytes, timestamp 2026-07-08 20:57:40.
- `supplemental_material.pdf`: generated successfully, 3 pages, 32727 bytes, timestamp 2026-07-08 20:57:39.

Log scan:

```powershell
Select-String -LiteralPath manuscript_v0.log,supplemental_material.log -Pattern 'Undefined|undefined|Fatal|Emergency|LaTeX Error|Citation.*undefined|Reference.*undefined|Label\(s\) may have changed|Rerun to get cross-references|There were undefined'
Select-String -LiteralPath manuscript_v0.log,supplemental_material.log -Pattern 'Overfull'
```

Outcome:

- No undefined references or citations were found.
- No fatal LaTeX errors were found.
- No overfull boxes were found.
- Underfull box warnings remain; these are non-blocking layout warnings for the current two-column draft.

## Packaging manifest changes made

- Updated `experiments_section_draft.tex` wording from "low-cost robustness slice" to "no-new-LLM robustness slice".
- Updated `submission_manifest.md` so the legacy-result scan targets active manuscript files, not the full draft directory where audit files intentionally preserve legacy values.
- This audit file should be kept as a submission-preparation artifact, not submitted as manuscript text.

## Final packaging checklist

Before actual submission:

- Replace or anonymize the author block in `front_matter_v1.tex`.
- Keep the 2Wiki table in the main PDF unless a target venue imposes a strict table limit.
- Run the active-file legacy scan from `submission_manifest.md` one more time.
- Recompile `manuscript_v0.tex` and `supplemental_material.tex` after any venue-template edits.
- Include `related_work_references.bib` or the generated `.bbl` according to the target venue instructions.
- Keep `model_invocation_ledger.md`, `reproducibility_protocol.md`, and `reviewer_response_bank.md` as internal/audit supplements unless the venue explicitly requests them.
