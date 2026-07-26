# Retrieval Paired Tests

This no-new-LLM analysis compares retrieval metrics over paired examples using frozen top-10 retrieval artifacts. Bootstrap intervals are computed over per-example paired deltas. McNemar's exact test is computed on full-support hit@10 transitions.

| Dataset | Baseline | Target | n | Baseline all-hit | Target all-hit | Delta all-hit [95% CI] | Baseline recall | Target recall | Delta recall [95% CI] | Full W->C / C->W | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense RAG | HyDE-style RAG | 500 | 0.6300 | 0.8680 | 0.2380 [0.2000, 0.2780] | 0.8080 | 0.9280 | 0.1200 [0.1010, 0.1410] | 120 / 1 | <0.0001 |
| HotpotQA | Single-query Reformulation RAG | HyDE-style RAG | 500 | 0.6180 | 0.8680 | 0.2500 [0.2140, 0.2860] | 0.7990 | 0.9280 | 0.1290 [0.1090, 0.1500] | 125 / 0 | <0.0001 |
| HotpotQA | Question + rewritten query | Question + hypothetical passage | 500 | 0.6280 | 0.8680 | 0.2400 [0.2020, 0.2780] | 0.8030 | 0.9280 | 0.1250 [0.1060, 0.1470] | 123 / 3 | <0.0001 |
| HotpotQA | Hybrid RAG | HyDE-style RAG | 500 | 0.7480 | 0.8680 | 0.1200 [0.0840, 0.1560] | 0.8690 | 0.9280 | 0.0590 [0.0410, 0.0780] | 76 / 16 | <0.0001 |
| HotpotQA | Iterative RAG | HyDE-style RAG | 500 | 0.7540 | 0.8680 | 0.1140 [0.0780, 0.1480] | 0.8700 | 0.9280 | 0.0580 [0.0390, 0.0780] | 72 / 15 | <0.0001 |
| 2WikiMultihopQA | Dense RAG | HyDE-style RAG | 500 | 0.3580 | 0.6680 | 0.3100 [0.2680, 0.3520] | 0.6695 | 0.8575 | 0.1880 [0.1665, 0.2085] | 157 / 2 | <0.0001 |
| 2WikiMultihopQA | BM25 RAG | HyDE-style RAG | 500 | 0.4320 | 0.6680 | 0.2360 [0.1940, 0.2780] | 0.7250 | 0.8575 | 0.1325 [0.1115, 0.1550] | 133 / 15 | <0.0001 |
| 2WikiMultihopQA | Hybrid RAG | HyDE-style RAG | 500 | 0.4480 | 0.6680 | 0.2200 [0.1800, 0.2620] | 0.7330 | 0.8575 | 0.1245 [0.1020, 0.1475] | 125 / 15 | <0.0001 |

W->C means the target retrieves all supporting titles when the baseline does not; C->W means the reverse.
