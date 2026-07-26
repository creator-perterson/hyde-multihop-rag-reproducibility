# Answer-absent Subset Paired Analysis

This diagnostic restricts each dataset to examples where the normalized gold-answer string does not appear in the generated HyDE hypothetical passage. It reuses existing retrieval and reader answer artifacts; no new LLM calls are made.

| Dataset | N | Dense all-hit | HyDE all-hit | Dense recall | HyDE recall | Dense F1 | HyDE F1 | Delta F1 [95% CI] | W->C | C->W | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 208 | 0.5817 | 0.7596 | 0.7812 | 0.8654 | 0.6110 | 0.6730 | 0.0620 [0.0170, 0.1080] | 20 | 8 | 0.0357 |
| 2WikiMultihopQA | 153 | 0.3856 | 0.6863 | 0.6797 | 0.8480 | 0.4058 | 0.6076 | 0.2019 [0.1372, 0.2671] | 30 | 3 | 0.0000 |

## Interpretation

A positive HyDE-over-Dense difference on this subset weakens the strict explanation that the observed HyDE gain is solely due to exact answer-string overlap in the hypothetical passage. This diagnostic does not prove the absence of parametric memory, paraphrased answer content, or benchmark contamination; it only removes exact normalized answer-string overlap from the analyzed subset.
