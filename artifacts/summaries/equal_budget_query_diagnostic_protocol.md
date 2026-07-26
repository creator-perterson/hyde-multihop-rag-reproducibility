# Equal-Budget Query-Composition Diagnostic Protocol

## Purpose

This diagnostic is designed to separate generated-text form from query-side call count and approximate generation budget. It compares four generated retrieval-text forms under the same model, one call per question, the same `max_tokens`, the same target output length, the same dense query serialization, the same BGE-base encoder, and the same top-k retrieval budget.

## Query Forms

- `keyword_expansion`: keyword/entity list.
- `direct_rewrite`: one single dense-retrieval rewrite.
- `question_decomposition`: numbered subquestions.
- `document_like_passage`: short encyclopedia-style passage.

All prompt rows target 40 words with a 35-45 word instruction and `max_tokens=64`.

## Completed Setup

Prompt artifacts have been generated under:

`local-artifacts/equal_budget_query_diagnostic/prompts`

The 100-example retrieval-only diagnostic has been completed for both datasets and all four query forms. Generation artifacts are stored under:

`local-artifacts/equal_budget_query_diagnostic/generations`

Each generation file contains 100 rows and 100 unique question ids. During generation, zero-byte files may appear while a process still has the output handle open; these are treated only as incomplete intermediate files. A batch is considered valid only after the process exits and the JSONL file passes row-count and unique-id validation.

Retrieval artifacts are stored under:

`local-artifacts/equal_budget_query_diagnostic/retrieval`

The current summaries are:

`local-artifacts/equal_budget_query_diagnostic/equal_budget_query_retrieval_summary.csv`

`local-artifacts/equal_budget_query_diagnostic/equal_budget_query_length_summary.csv`

Generation lengths are approximately matched across modes: HotpotQA mean word counts range from 38.17 to 39.36, and 2WikiMultihopQA mean word counts range from 38.03 to 39.87.

Retrieval-only results on the 100-example diagnostic subset are:

| Dataset | Query form | Any hit@10 | All-support hit@10 | Supporting-title recall@10 | Mean words |
|---|---:|---:|---:|---:|---:|
| HotpotQA | Keyword expansion | 1.0000 | 0.9600 | 0.9800 | 38.17 |
| HotpotQA | Direct rewrite | 0.9900 | 0.9100 | 0.9500 | 39.21 |
| HotpotQA | Question decomposition | 1.0000 | 0.8900 | 0.9450 | 39.36 |
| HotpotQA | Document-like passage | 1.0000 | 0.9700 | 0.9850 | 39.12 |
| 2WikiMultihopQA | Keyword expansion | 0.9900 | 0.7600 | 0.8825 | 38.03 |
| 2WikiMultihopQA | Direct rewrite | 0.9700 | 0.5900 | 0.7825 | 39.62 |
| 2WikiMultihopQA | Question decomposition | 1.0000 | 0.5200 | 0.7650 | 39.87 |
| 2WikiMultihopQA | Document-like passage | 1.0000 | 0.8300 | 0.9375 | 39.26 |

## Full Generation Commands

Run these only after approving the API cost. Full query generation is 4000 hosted LLM calls: 2 datasets x 4 query forms x 500 examples.

```powershell
$base = 'local-artifacts\equal_budget_query_diagnostic'
$pairs = @(
  @{ Dataset='ircot_hotpotqa_test500'; Prefix='ircot_hotpotqa_test500_equal_budget_bge_base' },
  @{ Dataset='ircot_2wikimultihopqa_test500'; Prefix='ircot_2wiki_test500_equal_budget_bge_base' }
)
$modes = @('keyword_expansion','direct_rewrite','question_decomposition','document_like_passage')
New-Item -ItemType Directory -Force -Path (Join-Path $base 'generations') | Out-Null
foreach ($pair in $pairs) {
  foreach ($mode in $modes) {
    $prompt = Join-Path $base "prompts\$($pair.Dataset)_${mode}_prompts.jsonl"
    $out = Join-Path $base "generations\$($pair.Prefix)_${mode}_generations.jsonl"
    python src\generator\generate_answers_openai_compatible.py `
      --prompts $prompt `
      --out $out `
      --model qwen3.7-max `
      --max_tokens 64 `
      --temperature 0.0 `
      --resume `
      --sleep 0.05
  }
}
```

## Retrieval Command

This step is local and uses cached BGE-base document embeddings when available.

```powershell
$env:PYTHONNOUSERSITE='1'
python src\retriever\run_equal_budget_query_retrieval.py `
  --datasets hotpotqa 2wiki `
  --generated_dir local-artifacts\equal_budget_query_diagnostic\generations `
  --results_dir local-artifacts\equal_budget_query_diagnostic\retrieval `
  --summary_csv local-artifacts\equal_budget_query_diagnostic\equal_budget_query_retrieval_summary.csv `
  --length_csv local-artifacts\equal_budget_query_diagnostic\equal_budget_query_length_summary.csv `
  --cache_dir local-artifacts\equal_budget_query_diagnostic\cache `
  --local_files_only `
  --doc_batch_size 64 `
  --query_batch_size 64
```

## Optional Reader Commands

Reader evaluation costs another 4000 hosted LLM calls if run for all four query forms on both datasets. Build reader prompts first, then call the hosted reader with `max_tokens=64`, `temperature=0.0`, and the same extractive prompt style used in the main experiments.

```powershell
$base = 'local-artifacts\equal_budget_query_diagnostic'
$pairs = @(
  @{ Prefix='ircot_hotpotqa_test500_equal_budget_bge_base' },
  @{ Prefix='ircot_2wiki_test500_equal_budget_bge_base' }
)
$modes = @('keyword_expansion','direct_rewrite','question_decomposition','document_like_passage')
New-Item -ItemType Directory -Force -Path (Join-Path $base 'reader_prompts') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $base 'reader_answers') | Out-Null
foreach ($pair in $pairs) {
  foreach ($mode in $modes) {
    $retrieval = Join-Path $base "retrieval\$($pair.Prefix)_${mode}_top10_retrieval.jsonl"
    $prompts = Join-Path $base "reader_prompts\$($pair.Prefix)_${mode}_top10_extractive_prompts.jsonl"
    $answers = Join-Path $base "reader_answers\$($pair.Prefix)_${mode}_top10_answers_qwenmax_500.jsonl"
    python src\generator\build_rag_prompts.py --retrieval $retrieval --out $prompts --top_k 10 --style extractive
    python src\generator\generate_answers_openai_compatible.py `
      --prompts $prompts `
      --out $answers `
      --model qwen3.7-max `
      --max_tokens 64 `
      --temperature 0.0 `
      --resume `
      --sleep 0.05
  }
}
```

## Interpretation Rules

If `document_like_passage` remains best after length matching, the paper can make a stronger mechanism claim about document-like organization. If the advantage shrinks or disappears, the paper should instead emphasize semantic information budget and avoid claiming an independent document-form effect.
