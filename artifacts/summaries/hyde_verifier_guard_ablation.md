# HyDE Verifier Guard Ablation

This no-new-LLM analysis reconstructs verifier variants from the same HyDE reader answers and the same risk-selected verifier outputs. It tests whether the final guard policy matters, without changing retrieval, reader prompts, or verifier generations.

| Variant | N | Verified | Changed | EM | F1 | Delta F1 | W->C | C->W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HyDE reader only | 500 | 0 | 0 | 0.6840 | 0.8105 | 0.0000 | 0 | 0 |
| HyDE + verifier raw | 500 | 120 | 5 | 0.6840 | 0.8119 | 0.0013 | 2 | 2 |
| HyDE + no-abstention guard | 500 | 120 | 5 | 0.6840 | 0.8119 | 0.0013 | 2 | 2 |
| HyDE + numeric-unit guard | 500 | 120 | 3 | 0.6880 | 0.8135 | 0.0030 | 2 | 0 |
| HyDE + both guards | 500 | 120 | 3 | 0.6880 | 0.8135 | 0.0030 | 2 | 0 |

## Interpretation

The raw verifier row applies the verifier final answer without post-processing. The no-abstention and numeric-unit rows isolate the two deterministic guards, and the both-guards row corresponds to the final conservative policy. The table should be read as a safety ablation: verification is intentionally a small canonicalization layer, and the guards are meant to prevent harmful transitions rather than to create the main performance gain.
