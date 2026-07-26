# HyDE Answer-Overlap Robustness Analysis

This analysis stratifies HotpotQA examples by whether the generated HyDE hypothetical passage contains the normalized gold answer string. It then compares Dense RAG and HyDE-style RAG within each group using existing answer and retrieval files; no new LLM calls are used.

| Group | n | Dense all-hit | HyDE all-hit | Delta all-hit | Dense F1 | HyDE F1 | Delta F1 | W->C | C->W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_examples | 500 | 0.6300 | 0.8680 | 0.2380 | 0.7199 | 0.8105 | 0.0906 | 53 | 8 |
| answer_in_hyde | 292 | 0.6644 | 0.9452 | 0.2808 | 0.7976 | 0.9085 | 0.1110 | 33 | 0 |
| answer_not_in_hyde | 208 | 0.5817 | 0.7596 | 0.1779 | 0.6110 | 0.6730 | 0.0620 | 20 | 8 |
| nontrivial_answer_in_hyde | 252 | 0.6825 | 0.9365 | 0.2540 | 0.8130 | 0.8953 | 0.0823 | 22 | 0 |

## Interpretation

The answer-not-in-HyDE group is the most important robustness slice. A positive HyDE gain in this group indicates that the method is not solely explained by explicit answer-string overlap in the hypothetical passage. The answer-in-HyDE group remains an answer-string-overlap-sensitive subgroup and should be interpreted as query-side answer-bearing expansion.
