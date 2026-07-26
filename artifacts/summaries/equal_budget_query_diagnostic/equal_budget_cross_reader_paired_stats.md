# Equal-budget Qwen-Turbo Reader Paired Statistics

This no-new-LLM diagnostic reuses fixed top-10 equal-budget BGE-base evidence and Qwen-Turbo answer files. Bootstrap intervals are paired 95% confidence intervals over per-example answer-score deltas. McNemar's exact test uses EM correctness transitions.

| Dataset | Baseline | Baseline EM/F1 | Doc-like EM/F1 | Delta EM | Delta F1 [95% CI] | W->C / C->W | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | Direct rewrite | 0.4960/0.6294 | 0.5660/0.7046 | 0.0700 | 0.0752 [0.0495, 0.1001] | 45 / 10 | <0.0001 |
| 2Wiki | Direct rewrite | 0.3460/0.4148 | 0.3620/0.4498 | 0.0160 | 0.0350 [0.0006, 0.0727] | 48 / 40 | 0.4557 |
| HotpotQA | Keyword/entity expansion | 0.5500/0.6777 | 0.5660/0.7046 | 0.0160 | 0.0269 [0.0053, 0.0503] | 24 / 16 | 0.2682 |
| 2Wiki | Keyword/entity expansion | 0.3640/0.4498 | 0.3620/0.4498 | -0.0020 | 0.0000 [-0.0354, 0.0342] | 36 / 37 | 1.0000 |
