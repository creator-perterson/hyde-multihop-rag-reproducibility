# Supporting Paragraph Metric Audit

| Dataset | Family | Method | Title all@10 | Paragraph all@10 | Paragraph recall@10 | Title-paragraph all gap | Paragraph all given title all |
|---|---|---|---:|---:|---:|---:|---:|
| HotpotQA | Main | Dense | 0.6300 | 0.6300 | 0.8080 | +0.0000 | 1.0000 |
| HotpotQA | Main | Multi-query | 0.6300 | 0.6300 | 0.8080 | +0.0000 | 1.0000 |
| HotpotQA | Main | BM25 | 0.7320 | 0.7320 | 0.8590 | +0.0000 | 1.0000 |
| HotpotQA | Main | Hybrid | 0.7480 | 0.7480 | 0.8690 | +0.0000 | 1.0000 |
| HotpotQA | Main | Rule-based iterative | 0.7540 | 0.7540 | 0.8700 | +0.0000 | 1.0000 |
| HotpotQA | Main | Direct rewrite | 0.6180 | 0.6180 | 0.7990 | +0.0000 | 1.0000 |
| HotpotQA | Main | Question + rewrite | 0.6280 | 0.6280 | 0.8030 | +0.0000 | 1.0000 |
| HotpotQA | Main | Hypothetical-only | 0.8480 | 0.8480 | 0.9180 | +0.0000 | 1.0000 |
| HotpotQA | Main | HyDE | 0.8680 | 0.8680 | 0.9280 | +0.0000 | 1.0000 |
| HotpotQA | Equal-budget | Keyword/entity | 0.9240 | 0.9240 | 0.9570 | +0.0000 | 1.0000 |
| HotpotQA | Equal-budget | Direct rewrite | 0.8800 | 0.8800 | 0.9330 | +0.0000 | 1.0000 |
| HotpotQA | Equal-budget | Decomposition | 0.8740 | 0.8740 | 0.9320 | +0.0000 | 1.0000 |
| HotpotQA | Equal-budget | Document-like | 0.9260 | 0.9260 | 0.9600 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Main | Dense | 0.3580 | 0.3580 | 0.6695 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Main | BM25 | 0.4320 | 0.4320 | 0.7250 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Main | Hybrid | 0.4480 | 0.4480 | 0.7330 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Main | Hypothetical-only | 0.6640 | 0.6640 | 0.8530 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Main | HyDE | 0.6680 | 0.6680 | 0.8575 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Equal-budget | Keyword/entity | 0.6760 | 0.6760 | 0.8410 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Equal-budget | Direct rewrite | 0.4760 | 0.4760 | 0.7190 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Equal-budget | Decomposition | 0.4540 | 0.4540 | 0.7295 | +0.0000 | 1.0000 |
| 2WikiMultihopQA | Equal-budget | Document-like | 0.7520 | 0.7520 | 0.9035 | +0.0000 | 1.0000 |

Paragraph metrics use exact normalized `(title, paragraph text)` keys for gold `is_supporting=true` corpus records.
In the released processed splits, each annotated supporting title is associated with one supporting paragraph record, so this audit primarily verifies exact processed-record recovery rather than introducing an independent sentence-level evidence notion.
Record metrics in the CSV additionally require the retrieved source-question-specific `doc_id`; this is stricter than evidence equivalence when the same title--paragraph text appears in multiple local pools.
