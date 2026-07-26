# Model Invocation Ledger

Date: 2026-07-25

This ledger freezes the model-facing settings and answer artifacts used by the current manuscript. It is not a provider-side request log: the generation wrapper stores model names, prompts, retrieved evidence, and predictions, but it does not store provider request IDs or per-request timestamps. Therefore, the dates below are local artifact freeze times from the generated JSONL files. They should be interpreted as the reproducible artifact window, not as proof of exact per-request API timing.

## What can and cannot be claimed

- Supported: the current manuscript uses frozen prompt and answer JSONL artifacts generated through an OpenAI-compatible chat-completions interface.
- Supported: reported generation used temperature-0 decoding (`temperature=0.0`) in the generation wrapper.
- Supported: configured output budgets are frozen separately by module: HyDE passage generation `max_tokens=64`, rewritten-query generation `max_tokens=64`, reader answer generation `max_tokens=64`, and verifier JSON generation `max_tokens=64`.
- Supported: the stored answer files retain the prompt, retrieved evidence, prediction, gold-answer metadata for evaluation, and model field when generated directly by the wrapper.
- Supported: canonical runs were completed inside a local frozen experiment window, and raw prompts, local raw text outputs, local artifact freeze timestamps, and stored model strings are retained because the hosted model alias may evolve.
- Supported: post-hoc equal-budget query-form generations use the stored API model string `qwen3.7-max` and are retained separately from the primary frozen scoring window.
- Supported: the targeted equal-budget reader robustness check uses frozen top-10 retrieval evidence and regenerates only reader answers with the stored API model string `qwen-turbo`.
- Not supported: exact provider-side request IDs, provider raw response envelopes, per-request provider timestamps, provider-side token usage by artifact, wall-clock latency, or billing cost.
- Not supported: exact replay of a provider-internal model snapshot when hosted aliases such as `qwen3.7-max` or `qwen-turbo` change.
- Dashboard context: the user-provided provider console screenshot shows a one-week usage view with Qwen3.7-Max as the dominant model, including 6,004 Qwen3.7-Max calls, 7,820 total successful calls, 13,946K total tokens, and 1,783 average tokens per request. This is useful as an external usage sanity check, but it is an aggregate dashboard view rather than a per-artifact audit export.

## API and decoding settings

| Field | Frozen value |
|---|---|
| Provider | Alibaba Cloud Bailian/DashScope |
| API interface | OpenAI-compatible chat completions |
| Endpoint type | OpenAI-compatible `/v1` chat-completions endpoint, loaded from local `LLM_BASE_URL` |
| Endpoint region | Not exported into local JSONL artifacts; not claimed as reproducible |
| Generation script | `src\generator\generate_answers_openai_compatible.py` |
| Model family reported in manuscript | Qwen3.7-Max for the primary pipeline and equal-budget query-form generation; Qwen-Turbo for the targeted reader robustness check |
| API model string for canonical runs | `qwen3.7-max` for the primary pipeline; post-hoc equal-budget query-form generation also uses `qwen3.7-max`; targeted equal-budget reader check uses `qwen-turbo` |
| Stored model fields | Mostly `qwen3.7-max`; some early baseline files store dated aliases such as `qwen3.7-max-2026-05-17`, `qwen3.7-max-2026-05-20`, or `qwen3.7-max-2026-06-08`; targeted reader-check files store `qwen-turbo` |
| Immutable snapshot/version | No provider-side immutable snapshot ID is stored or publicly available in these artifacts |
| Temperature | `0.0` |
| Top-p | Not explicitly sent by the wrapper; provider default applies |
| HyDE passage generation budget | `max_tokens=64` |
| Rewritten-query generation budget | `max_tokens=64` |
| Reader answer budget | `max_tokens=64` |
| Verifier JSON budget | `max_tokens=64` |
| Thinking/reasoning mode | Not explicitly enabled or parameterized; standard chat-completions mode is used |
| Seed/logprobs | Not sent |
| Timeout | `60` seconds unless overridden by environment |
| Retries | `3` unless overridden by environment |
| Retry sleep | `5` seconds unless overridden by environment |
| Resume behavior | `--resume` skips completed IDs already present in the output JSONL and appends only missing rows |
| Failure replacement | Canonical `qwen3.7-max` HyDE, matched-call, 2Wiki, and verifier runs, as well as the `qwen-turbo` reader check, use completed rows from frozen output files. The early LLM-only baseline includes 5 fallback rows from a dated Qwen3.7-Max alias and is disclosed separately. |
| API credentials | Loaded from local environment variables; not stored or shared |
| Prompt retention | Full prompt text retained in answer JSONL files |
| Raw output retention | Prediction text is retained; provider response envelopes, request IDs, token usage, per-request timestamps, and latency are not retained |
| Query/reader model consistency | Canonical HyDE generation, matched-call query reformulation, reader, 2Wiki, and verifier artifacts use the same API model string, `qwen3.7-max`; post-hoc equal-budget query-form generation also uses `qwen3.7-max` but is not part of the primary frozen scoring window; the targeted equal-budget reader check intentionally changes only the reader to `qwen-turbo` over fixed retrieved evidence; because these are hosted aliases, the ledger does not claim an immutable provider-internal snapshot across the full window |
| Temperature-0 replay boundary | Temperature 0 is used to reduce sampling variation, but it is not claimed to guarantee byte-identical outputs across requests or across hosted-model alias updates |
| Gold-answer handling | Gold-answer metadata retained for scoring, but not inserted into model prompt text |
| JSON parsing | Reader and HyDE generations are stored as raw text. Verifier outputs are parsed as JSON when possible, with substring JSON extraction for wrapped responses; final reporting uses deterministic guarded merge rules. |

## Prompt and artifact hashes

These hashes freeze the prompt-generation code and representative canonical prompt files used by the main HyDE and verifier pipeline.

| Artifact | SHA-256 |
|---|---|
| `src\generator\build_hyde_prompts.py` | `89ECB4FBF56C26F3FDDA69E4C78848D3EBD30C912D2B266AFA9B6BA4AE521858` |
| `src\generator\build_rag_prompts.py` | `98904C375E456F7AF1D63377F44436465882B20B0247784E3FD8D1C3B4A44584` |
| `src\generator\build_query_reformulation_prompts.py` | `FF9EA7EF3BAFA5D3284F12572588755F20C33894333DBDD97CCF0BAC3424952C` |
| `src\verifier\build_verification_prompts.py` | `BB2495BC4899F596E7926FF3686F5DAAC1F522A28AE590638C8F5D7112581FA9` |
| `local-artifacts\results\ircot_hotpotqa_test500_hyde_generation_prompts.jsonl` | `4D43047C6E313B4206BACE983B98EF7A81B0E8C9E7330345117C5BE2C21BB5B2` |
| `local-artifacts\results\ircot_hotpotqa_test500_hyde_top10_extractive_prompts.jsonl` | `DCBC0B3E9E415B3EE5966D880CFFD2B010384A27B5733341D9DC426A02CACF96` |
| `local-artifacts\results\qwenmax_hyde_verification_prompts_conservative_risk120.jsonl` | `763D188CAF8DBCB70B17A28D37DF0F7CE9734C57C6E0E17B5A2C0FCC33F3D618` |

## HotpotQA main artifacts

| Manuscript role | Artifact | Rows | Stored model field(s) observed | Local freeze time |
|---|---|---:|---|---|
| LLM-only baseline | `local-artifacts\results\ircot_hotpotqa_test500_llm_only_answers_qwen_500.jsonl` | 500 | `qwen3.7-max`; `qwen3.7-max-2026-05-20` | 2026-07-03 19:31:10 |
| One-step Dense RAG | `local-artifacts\results\ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl` | 500 | `qwen3.7-max-2026-05-17`; `qwen3.7-max-2026-06-08` | 2026-07-02 15:24:45 |
| BM25 RAG | `local-artifacts\results\ircot_hotpotqa_test500_bm25_top10_extractive_answers_qwen_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-02 21:15:15 |
| BM25 + Dense Hybrid | `local-artifacts\results\ircot_hotpotqa_test500_hybrid_top10_extractive_answers_qwen_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-03 11:44:06 |
| Multi-query RAG | `local-artifacts\results\ircot_hotpotqa_test500_multiquery_top10_decay050_extractive_answers_qwen_500.jsonl` | 500 | `qwen3.7-max-2026-05-20` | 2026-07-04 10:30:08 |
| Single-query reformulation generation | `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-08 19:45:29 |
| Single-query Reformulation RAG reader | `local-artifacts\results\ircot_hotpotqa_test500_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-08 20:16:05 |
| Question + Single-query Reformulation RAG reader | `local-artifacts\results\ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-09 10:20:58 |
| Evidence-guided Iterative RAG canonical rerun | `local-artifacts\results\ircot_hotpotqa_test500_iterative_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-04 12:54:48 |
| HyDE hypothetical-passage generation | `local-artifacts\results\ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-04 11:52:59 |
| HyDE-style RAG reader | `local-artifacts\results\ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-04 12:14:39 |
| HyDE hypothetical-only mechanism reader | `local-artifacts\results\ircot_hotpotqa_test500_hyde_hypothetical_only_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-06 18:29:43 |
| HyDE conservative verifier calls | `local-artifacts\results\qwenmax_hyde_verification_answers_conservative_risk120.jsonl` | 120 | `qwen3.7-max` | 2026-07-04 13:24:27 |
| HyDE + conservative verifier merged result | `local-artifacts\results\qwenmax_hyde_selective_verification_eval_risk120_numeric_guarded_500.jsonl` | 500 | merged evaluation file; no direct model field | 2026-07-04 13:28:08 |

## 2WikiMultihopQA secondary artifacts

| Manuscript role | Artifact | Rows | Stored model field(s) observed | Local freeze time |
|---|---|---:|---|---|
| 2Wiki Dense RAG reader | `local-artifacts\results\ircot_2wiki_test500_dense_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-06 09:20:08 |
| 2Wiki BM25 RAG reader | `local-artifacts\results\ircot_2wiki_test500_bm25_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-06 15:55:31 |
| 2Wiki Hybrid RAG reader | `local-artifacts\results\ircot_2wiki_test500_hybrid_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-06 09:52:45 |
| 2Wiki HyDE hypothetical-passage generation | `local-artifacts\results\ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-05 16:01:42 |
| 2Wiki HyDE-style RAG reader | `local-artifacts\results\ircot_2wiki_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl` | 500 | `qwen3.7-max` | 2026-07-05 16:34:21 |

## Equal-budget query-form generation artifacts

These are post-hoc robustness diagnostics added after the primary HotpotQA analysis to test query-form and generation-budget explanations. The prompt family was fixed once, with a 40-word target, a 35-45 word instruction window, and a 64-token generation cap, then applied to both HotpotQA and 2Wiki without dataset-specific prompt edits.

| Manuscript role | Artifact | Rows | Stored model field(s) observed | Local freeze time |
|---|---|---:|---|---|
| HotpotQA direct-rewrite equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_hotpotqa_test500_equal_budget_bge_base_direct_rewrite_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-22 16:31:31 |
| HotpotQA keyword/entity equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_hotpotqa_test500_equal_budget_bge_base_keyword_expansion_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-22 14:25:43 |
| HotpotQA question-decomposition equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_hotpotqa_test500_equal_budget_bge_base_question_decomposition_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-22 14:55:15 |
| HotpotQA document-like equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_hotpotqa_test500_equal_budget_bge_base_document_like_passage_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-22 13:51:15 |
| 2Wiki direct-rewrite equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_2wiki_test500_equal_budget_bge_base_direct_rewrite_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-23 12:26:00 |
| 2Wiki keyword/entity equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_2wiki_test500_equal_budget_bge_base_keyword_expansion_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-23 11:08:59 |
| 2Wiki question-decomposition equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_2wiki_test500_equal_budget_bge_base_question_decomposition_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-23 11:52:06 |
| 2Wiki document-like equal-budget generation | `local-artifacts\equal_budget_query_diagnostic\generations\ircot_2wiki_test500_equal_budget_bge_base_document_like_passage_generations.jsonl` | 500 | `qwen3.7-max` | 2026-07-23 10:50:54 |

## Equal-budget Qwen-Turbo reader robustness artifacts

| Manuscript role | Artifact | Rows | Stored model field(s) observed | Local freeze time |
|---|---|---:|---|---|
| HotpotQA direct-rewrite evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_hotpotqa_test500_equal_budget_bge_base_direct_rewrite_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-23 15:01:41 |
| HotpotQA keyword/entity-expansion evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_hotpotqa_test500_equal_budget_bge_base_keyword_expansion_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-25 17:54:23 |
| HotpotQA document-like evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_hotpotqa_test500_equal_budget_bge_base_document_like_passage_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-23 15:01:41 |
| 2Wiki direct-rewrite evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_2wiki_test500_equal_budget_bge_base_direct_rewrite_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-23 15:05:10 |
| 2Wiki keyword/entity-expansion evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_2wiki_test500_equal_budget_bge_base_keyword_expansion_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-25 17:54:20 |
| 2Wiki document-like evidence reader | `local-artifacts\equal_budget_query_diagnostic\reader_answers\ircot_2wiki_test500_equal_budget_bge_base_document_like_passage_top10_answers_qwenturbo_500.jsonl` | 500 | `qwen-turbo` | 2026-07-23 15:05:13 |

## Reproducibility boundary

The current files are sufficient to re-score the reported EM/F1 results and to audit the exact prompt text used for the frozen runs. They are not sufficient to reproduce the provider's internal model snapshot if hosted aliases such as `qwen3.7-max` or `qwen-turbo` change. If a provider usage CSV or request-level export becomes available, append it to this ledger and cite it as provider-side evidence rather than replacing the frozen local artifact records above.
