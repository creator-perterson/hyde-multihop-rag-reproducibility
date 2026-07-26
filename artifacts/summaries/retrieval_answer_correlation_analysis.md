# Retrieval-Answer Correlation Analysis

This analysis examines whether supporting-title completeness is associated with answer quality on the IRCoT HotpotQA test_subsampled split.

## Method-Level Correlation

| Correlation | Pearson r |
| --- | ---: |
| all_hit_vs_em | 0.9931 |
| all_hit_vs_f1 | 0.9828 |
| mean_recall_vs_em | 0.9942 |
| mean_recall_vs_f1 | 0.9834 |

## Supporting-Title-Completeness Buckets

| Method | Supporting-title bucket | n | EM | F1 | Mean title recall |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 + Dense Hybrid | full title support | 374 | 0.7273 | 0.8616 | 1.0000 |
| BM25 + Dense Hybrid | partial title support | 121 | 0.3967 | 0.4677 | 0.5000 |
| BM25 + Dense Hybrid | no title support | 5 | 0.0000 | 0.0000 | 0.0000 |
| BM25 RAG | full title support | 366 | 0.7268 | 0.8546 | 1.0000 |
| BM25 RAG | partial title support | 127 | 0.3622 | 0.4615 | 0.5000 |
| BM25 RAG | no title support | 7 | 0.0000 | 0.0571 | 0.0000 |
| Evidence-guided Iterative Retrieval | full title support | 377 | 0.7215 | 0.8611 | 1.0000 |
| Evidence-guided Iterative Retrieval | partial title support | 116 | 0.3966 | 0.4623 | 0.5000 |
| Evidence-guided Iterative Retrieval | no title support | 7 | 0.0000 | 0.0000 | 0.0000 |
| HyDE-style RAG | full title support | 434 | 0.7396 | 0.8720 | 1.0000 |
| HyDE-style RAG | partial title support | 60 | 0.3500 | 0.4472 | 0.5000 |
| HyDE-style RAG | no title support | 6 | 0.0000 | 0.0000 | 0.0000 |
| Multi-query RAG | full title support | 315 | 0.7143 | 0.8457 | 1.0000 |
| Multi-query RAG | partial title support | 178 | 0.3764 | 0.4721 | 0.5000 |
| Multi-query RAG | no title support | 7 | 0.1429 | 0.1429 | 0.0000 |
| One-step Dense RAG | full title support | 315 | 0.7079 | 0.8495 | 1.0000 |
| One-step Dense RAG | partial title support | 178 | 0.4045 | 0.5077 | 0.5000 |
| One-step Dense RAG | no title support | 7 | 0.2857 | 0.2857 | 0.0000 |

## Interpretation

Across retrieval-based methods, higher all-support hit@10 and mean supporting-title recall are positively associated with answer EM/F1. Because the method-level correlation is computed over only six canonical retrieval-only variants, it should be interpreted as descriptive evidence rather than as a standalone causal proof. At the sample level, examples with all supporting titles retrieved consistently achieve substantially higher answer quality than partial-title-support or no-title-support examples. Together, these two analyses support the more conservative paper claim that supporting-title completeness, aligned with paragraph-text recovery in the processed artifacts, is a central bottleneck in this lightweight multi-hop RAG setting.

Verifier-only outputs and LLM-only/no-retrieval outputs are not included in the correlation tables because the goal here is to isolate the relationship between supporting-evidence acquisition and answer quality. Verification is analyzed separately as a conservative safety layer.

Primary retrieval strategies included in method-level correlation: 6.
