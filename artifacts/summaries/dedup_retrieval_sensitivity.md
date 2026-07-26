# Exact Title-Paragraph Deduplication Sensitivity

| Dataset | Method | Retain all@10 | Dedup all@10 | Delta all | Retain recall | Dedup recall | Delta recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | Dense | 0.6300 | 0.6320 | +0.0020 | 0.8080 | 0.8090 | +0.0010 |
| HotpotQA | Hybrid | 0.7480 | 0.7520 | +0.0040 | 0.8690 | 0.8710 | +0.0020 |
| HotpotQA | HyDE | 0.8680 | 0.8680 | +0.0000 | 0.9280 | 0.9280 | +0.0000 |
| 2WikiMultihopQA | Dense | 0.3580 | 0.3920 | +0.0340 | 0.6695 | 0.6920 | +0.0225 |
| 2WikiMultihopQA | Hybrid | 0.4480 | 0.4600 | +0.0120 | 0.7330 | 0.7415 | +0.0085 |
| 2WikiMultihopQA | HyDE | 0.6680 | 0.6760 | +0.0080 | 0.8575 | 0.8625 | +0.0050 |

The deduplicated index removes exact normalized `(title, paragraph text)` duplicates before retrieval.
Reader answers are not regenerated; this is a retrieval-side sensitivity check.
