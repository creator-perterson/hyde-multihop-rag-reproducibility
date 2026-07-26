# Reproducibility Protocol

Date: 2026-07-08

This protocol records the current reproducible experiment path for the released IRCoT HotpotQA and 2WikiMultihopQA `test_subsampled` settings. It is intended to support the Experiments section, document prompt/model/settings choices, and prevent result-file drift.

## Environment

```powershell
conda activate agentic_rag
cd <repo-root>
```

Required Python packages are recorded in:

```text
<repo-root>\requirements.txt
```

LLM API settings are loaded from:

```text
<repo-root>\.env
```

Do not commit or share `.env`.

## Data

Released IRCoT processed data:

```text
external-data\ircot\processed_data\hotpotqa\test_subsampled.jsonl
```

Converted local dataset:

```text
local-artifacts\datasets\ircot_hotpotqa_test500\questions.jsonl
local-artifacts\datasets\ircot_hotpotqa_test500\corpus.jsonl
local-artifacts\datasets\ircot_2wikimultihopqa_test500\questions.jsonl
local-artifacts\datasets\ircot_2wikimultihopqa_test500\corpus.jsonl
```

Conversion command:

```powershell
python .\src\data\prepare_ircot_hotpotqa.py --input ..\_external\paper_repos\ircot\processed_data\hotpotqa\test_subsampled.jsonl --out_dir datasets\ircot_hotpotqa_test500
```

Corpus-statistics command:

```powershell
python .\src\evaluation\analyze_corpus_statistics.py --code_root . --out_csv local-artifacts\corpus_statistics_test500.csv --out_tex paper\latex\table_corpus_statistics.tex
```

Deduplication sensitivity command:

```powershell
conda run -n agentic_rag python .\src\evaluation\analyze_dedup_retrieval_sensitivity.py --code_root . --out_csv local-artifacts\dedup_retrieval_sensitivity.csv --out_md local-artifacts\dedup_retrieval_sensitivity.md --out_tex paper\latex\table_dedup_retrieval_sensitivity.tex
```

## Retrieval Index

Build FAISS index:

```powershell
python .\src\retriever\build_faiss_index.py --corpus datasets\ircot_hotpotqa_test500\corpus.jsonl --out_dir datasets\ircot_hotpotqa_test500\faiss_index --model sentence-transformers/all-MiniLM-L6-v2
```

Index settings:

| Item | Value |
|---|---|
| Encoder | sentence-transformers/all-MiniLM-L6-v2 |
| Index | FAISS IndexFlatIP |
| Embedding normalization | yes |
| Document text | title + paragraph text |

## One-Step Dense RAG

Retrieve top-10:

```powershell
python .\src\retriever\retrieve_topk.py --questions datasets\ircot_hotpotqa_test500\questions.jsonl --index_dir datasets\ircot_hotpotqa_test500\faiss_index --out results\ircot_hotpotqa_test500_top10_retrieval.jsonl --top_k 10
```

Evaluate retrieval:

```powershell
python .\src\evaluation\evaluate_retrieval.py --retrieval results\ircot_hotpotqa_test500_top10_retrieval.jsonl
```

Expected retrieval output:

```text
Questions: 500
Any supporting-title hit@k: 0.9860
All supporting-title hit@k: 0.6300
Mean supporting-title recall@k: 0.8080
```

Build prompts:

```powershell
python .\src\generator\build_rag_prompts.py --retrieval results\ircot_hotpotqa_test500_top10_retrieval.jsonl --out results\ircot_hotpotqa_test500_top10_extractive_prompts.jsonl --top_k 10 --style extractive --max_chars_per_doc 900
```

Generate answers:

```powershell
python .\src\generator\generate_answers_openai_compatible.py --prompts results\ircot_hotpotqa_test500_top10_extractive_prompts.jsonl --out results\ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl --resume
```

Expected answer result:

```text
EM 0.5940
F1 0.7199
```

## Evidence-Guided Iterative Retrieval

Run iterative retrieval:

```powershell
python .\src\retriever\retrieve_iterative_topk.py --questions datasets\ircot_hotpotqa_test500\questions.jsonl --index_dir datasets\ircot_hotpotqa_test500\faiss_index --first_retrieval results\ircot_hotpotqa_test500_top10_retrieval.jsonl --out results\ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl --top_k 10 --per_query_k 10 --expand_docs 3
```

Evaluate retrieval:

```powershell
python .\src\evaluation\evaluate_retrieval.py --retrieval results\ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl
```

Expected retrieval output:

```text
Questions: 500
Any supporting-title hit@k: 0.9860
All supporting-title hit@k: 0.7540
Mean supporting-title recall@k: 0.8700
```

Build prompts:

```powershell
python .\src\generator\build_rag_prompts.py --retrieval results\ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl --out results\ircot_hotpotqa_test500_iterative_top10_extractive_prompts.jsonl --top_k 10 --style extractive --max_chars_per_doc 900
```

Generate answers:

```powershell
python .\src\generator\generate_answers_openai_compatible.py --prompts results\ircot_hotpotqa_test500_iterative_top10_extractive_prompts.jsonl --out results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl --resume
```

Canonical answer result:

```text
EM 0.6360
F1 0.7565
```

The earlier `ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwen_500.jsonl` file reached EM/F1 = 0.6660/0.7894 with a legacy `qwen3.7-max-2026-05-17` reader snapshot. The main paper now uses the aligned `qwen3.7-max` rerun at `results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl`.

## Selective Verification Agent

Build Expanded120 conservative verification prompts:

```powershell
python .\src\verifier\build_verification_prompts.py --answers results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl --out results\qwenmax_iterative_verification_prompts_conservative_risk120.jsonl --sample_strategy risk --risk_target 120 --top_k 10 --style conservative --max_chars_per_doc 900
```

Generate verifier answers:

```powershell
python .\src\generator\generate_answers_openai_compatible.py --prompts results\qwenmax_iterative_verification_prompts_conservative_risk120.jsonl --out results\qwenmax_iterative_verification_answers_conservative_risk120.jsonl --resume
```

Merge with guarded post-processing:

```powershell
python .\src\verifier\evaluate_selective_verification.py --base_answers results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl --verifier_answers results\qwenmax_iterative_verification_answers_conservative_risk120.jsonl --out_jsonl results\qwenmax_iterative_selective_verification_eval_risk120_500.jsonl
```

Expected merged result:

```text
Questions: 500
Verified subset: 120
Initial EM: 0.6360
Final EM: 0.6400
Initial F1: 0.7565
Final F1: 0.7595
Verified-subset transitions:
  correct_to_correct: 61
  wrong_to_wrong: 57
  wrong_to_correct: 2
```

## Analysis Commands

Pairwise dense vs iterative comparison:

```powershell
python .\src\evaluation\compare_answer_sets.py --dense results\ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl --iterative results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl --out_csv results\stage1_dense_vs_iterative_qwenmax_comparison.csv --out_jsonl results\stage1_dense_vs_iterative_qwenmax_comparison.jsonl
```

Expected output:

```text
Compared questions: 500
Dense EM/F1: 0.5940 / 0.7199
Iterative EM/F1: 0.6360 / 0.7565
Delta F1: 0.0366
Categories:
  both_correct: 285
  both_wrong: 170
  dense_only_correct: 12
  iterative_only_correct: 33
```

Error analysis:

```powershell
python .\src\evaluation\analyze_qwen_errors.py --answers results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl --out_jsonl results\qwen_iterative_error_cases.jsonl --out_md results\qwen_iterative_error_analysis.md
```

Expected output:

```text
Analyzed examples: 500
correct: 333
answer_format_or_alias: 101
retrieval_miss: 47
reader_reasoning_error: 19
```

## Legacy Result Files From Earlier Iterative-RAG Draft

The following entries are retained only as historical context from the earlier iterative-RAG draft. The current manuscript should use the `Current Canonical Ledger` section below when the two sections conflict.

| Legacy manuscript item | Source file |
|---|---|
| Legacy main-results table | `local-artifacts\ircot_test500_qwen_results.csv` |
| Legacy iterative verifier ablation | `local-artifacts\Answer_Verification_Agent_实验记录.md` |
| Dense answers | `local-artifacts\results\ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl` |
| Iterative answers | `local-artifacts\results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwen_500.jsonl` |
| Best verifier merge | `local-artifacts\results\qwen_iterative_selective_verification_eval_expanded120_v2_guarded_numeric_500.jsonl` |
| Pairwise analysis | `local-artifacts\results\qwen_pairwise_comparison.csv` |
| Error analysis | `local-artifacts\results\qwen_iterative_error_analysis.md` |

## Current Canonical Ledger

The current manuscript uses the following canonical artifacts. This section supersedes older iterative-only notes above when there is a conflict.

| Manuscript item | Source file |
|---|---|
| Main HotpotQA results | `local-artifacts\ircot_test500_qwen_results.csv` |
| Stage 1 canonical ledger audit | `local-artifacts\stage1_experiment_result_audit_2026-07-06.md` |
| HyDE mechanism ablation | `local-artifacts\hyde_mechanism_ablation_hotpotqa_test500.csv` |
| Local candidate-corpus statistics | `src\evaluation\analyze_corpus_statistics.py`; `local-artifacts\corpus_statistics_test500.csv`; `paper\latex\table_corpus_statistics.tex` |
| Deduplication sensitivity | `src\evaluation\analyze_dedup_retrieval_sensitivity.py`; `local-artifacts\dedup_retrieval_sensitivity.csv`; `local-artifacts\dedup_retrieval_sensitivity.md`; `paper\latex\table_dedup_retrieval_sensitivity.tex` |
| HyDE answer-string-overlap audit | `local-artifacts\hyde_leakage_audit_hotpotqa_test500.md` |
| HyDE answer-overlap robustness | `local-artifacts\hyde_overlap_robustness_hotpotqa_test500.csv`; `local-artifacts\hyde_overlap_robustness_hotpotqa_test500.md` |
| Answer-absent paired CI diagnostic | `local-artifacts\answer_absent_subset_paired_ci.csv`; `local-artifacts\answer_absent_subset_paired_ci.md`; `paper\latex\table_answer_absent_subset.tex` |
| Retrieval-side paired tests | `src\evaluation\analyze_retrieval_paired_tests.py`; `local-artifacts\retrieval_paired_tests.csv`; `local-artifacts\retrieval_paired_tests.md`; `paper\latex\table_retrieval_paired_tests.tex` |
| Statistical reliability tables | `local-artifacts\hotpotqa_test500_method_bootstrap_ci.csv`; `local-artifacts\hotpotqa_test500_paired_method_deltas.csv` |
| Evidence-completeness buckets | `local-artifacts\retrieval_answer_correlation_by_bucket.csv`; `paper\latex\table_evidence_completeness_buckets_supp.tex` |
| Efficiency profile | `local-artifacts\lightweight_efficiency_profile_hotpotqa_test500.csv` |
| Configuration provenance table | `paper\latex\table_configuration_provenance.tex` |
| HyDE verifier guard ablation | `local-artifacts\hyde_verifier_guard_ablation.csv`; `local-artifacts\hyde_verifier_guard_ablation.md`; `paper\latex\table_hyde_verifier_guard_ablation.tex` |
| Artifact-grounded qualitative cases | `src\evaluation\extract_case_study_examples.py`; `local-artifacts\artifact_grounded_case_studies.md`; `paper\latex\table_qualitative_case_study.tex` |
| Secondary 2Wiki results | `local-artifacts\2WikiMultihopQA_test500_migration_log.md`; `paper\latex\table_2wiki_generalization.tex` |
| Supplemental 2Wiki HyDE answer-string-overlap audit | `local-artifacts\2wiki_hyde_leakage_audit_test500.md`; `local-artifacts\results\ircot_2wiki_test500_hyde_leakage_audit.jsonl` |
| Supplemental 2Wiki HyDE hypothetical-only retrieval | `local-artifacts\results\ircot_2wiki_test500_hyde_hypothetical_only_top10_retrieval.jsonl`; `paper\latex\table_2wiki_hyde_query_mechanism.tex` |
| Model invocation ledger | `docs\model_invocation_ledger.md`; optional table `paper\latex\table_model_invocation_ledger.tex` |
| Submission package guard | `paper\latex\submission_manifest.md` |
| Dense answers | `local-artifacts\results\ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl` |
| Question + single-query reformulation control | `local-artifacts\results\ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_retrieval.jsonl`; `local-artifacts\results\ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl` |
| Canonical iterative answers | `local-artifacts\results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl` |
| HyDE answers | `local-artifacts\results\ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl` |
| Best HyDE verifier merge | `local-artifacts\results\qwenmax_hyde_selective_verification_eval_risk120_numeric_guarded_500.jsonl` |
| Error analysis | `local-artifacts\error_analysis_hyde_verifier.md` |

## Statistical Analysis Settings

Paired confidence intervals are computed by resampling question IDs with replacement over paired per-question deltas. The canonical analysis scripts use `B=2000` bootstrap replicates and fixed random seed `13`; reported intervals are two-sided 95% percentile intervals. Paired EM comparisons use the two-sided exact McNemar test over discordant outcomes, implemented as an exact binomial tail calculation in `src\evaluation\analyze_retrieval_answer_correlation.py`.

## Prompt, Model, and Decoding Settings

The reported generation runs use fixed prompt templates from:

```text
src\generator\build_hyde_prompts.py
src\generator\build_rag_prompts.py
src\verifier\build_verification_prompts.py
```

Reader and verifier evidence blocks use `--top_k 10` and `--max_chars_per_doc 900`. The generation wrapper is:

```text
src\generator\generate_answers_openai_compatible.py
```

Reported settings:

| Item | Value |
|---|---|
| Provider | Alibaba Cloud Bailian/DashScope |
| API interface | OpenAI-compatible `/v1` chat completions |
| Main reader/generator model | Qwen3.7-Max |
| Canonical API model string | `qwen3.7-max` |
| Immutable snapshot/version | Not available in local artifacts; hosted alias may evolve |
| Endpoint region | Not exported into local JSONL artifacts |
| Temperature | 0.0 |
| Top-p | Not explicitly sent; provider default applies |
| Max answer tokens | 64 |
| Thinking/reasoning mode | Not explicitly enabled or parameterized |
| Timeout default | 60 seconds |
| Retry default | 3 attempts |
| Retry sleep default | 5 seconds |
| Failure handling | Requests are retried; `--resume` skips completed IDs and appends missing rows. Canonical rows come from frozen output files; the early LLM-only baseline's 5 fallback rows are disclosed in the model invocation ledger. |
| JSON handling | Reader and HyDE outputs are stored as raw text predictions. Verifier outputs are parsed as JSON when possible and then merged with deterministic guards. |
| Stored answer fields | id, question, gold-answer metadata, retrieved evidence, prompt, prediction, model |
| Raw output boundary | Prompt text and prediction text are retained; provider response envelopes, request IDs, token usage, latency, and billing are not retained |

Gold-answer metadata is retained in JSONL rows for evaluation and error analysis, but the prompt text passed to the model does not include the gold answer.

The model invocation ledger in `docs\model_invocation_ledger.md` records the local freeze window, stored model fields, decoding settings, prompt hashes, and exact prompt/answer artifacts for Qwen3.7-Max calls. The ledger should be read as a frozen artifact record rather than as a provider-side request log, because the generation wrapper does not store request IDs, per-request timestamps, provider-side token usage, latency, or billing information. All canonical runs were completed within a frozen experiment window, and raw prompts and local outputs are retained because the hosted model alias may evolve.

## Reviewer-Facing Boundary Checks

The manuscript should preserve these seven defenses:

1. HyDE itself is not claimed as the new method; the contribution is the controlled lightweight multi-hop adaptation, mechanism audit, reliability analysis, efficiency profile, and conservative verifier design.
2. The generated hypothetical passage is query expansion only; it is not final-reader evidence.
3. Multi-query and Dense retrieval metrics are identical because the fused top-10 document sets were identical under the selected setting.
4. The main table uses the canonical Qwen3.7-Max iterative rerun, not the legacy iterative reader snapshot.
5. Pearson correlations are descriptive and are supported by answer-side paired bootstrap tests, retrieval-side paired tests, McNemar transitions, and evidence-completeness buckets.
6. Lightweight cost is supported by call counts, prompt/evidence character counts, verifier coverage, and no-training status.
7. HyDE generation can contain answer-like strings, so the method is described as generative query expansion rather than retrieval free of answer-string overlap. Answer-absent paired slices on HotpotQA and 2Wiki reduce the sufficiency of an exact-string-overlap explanation but do not prove absence of paraphrased answer content or model memory.
8. The verifier is a small guarded canonicalization layer. The HyDE guard ablation shows that raw verifier outputs can introduce correct-to-wrong transitions and that the final numeric guard removes the observed harmful transitions.

## Current Boundary

This protocol is a processed-data reproducibility protocol. It should not be described as a full-wiki IRCoT reproduction. If a future experiment uses full-wiki retrieval or the original IRCoT retrieval stack, this protocol and the Experiments section should be updated.
