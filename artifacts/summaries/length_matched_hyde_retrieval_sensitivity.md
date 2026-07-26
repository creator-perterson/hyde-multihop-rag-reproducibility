# Query Length and Length-matched HyDE Retrieval Sensitivity

This no-new-LLM analysis uses frozen HotpotQA query-generation artifacts. Token counts use the MiniLM tokenizer. The hypothetical-only control truncates each hypothetical passage to the corresponding rewritten-query token count. The question-plus-hypothetical control truncates only the hypothetical passage so that the serialized query has no more MiniLM tokens than the corresponding question-plus-rewrite query.

## Query Token Lengths

| Query | Mean tokens | Median | P95 | Hit 256-token cap |
|---|---:|---:|---:|---:|
| q | 19.9140 | 19.0000 | 33 | 0 |
| r | 10.9380 | 10.0000 | 17 | 0 |
| q+r | 35.8520 | 34.0000 | 52 | 0 |
| h | 68.7400 | 68.0000 | 92 | 0 |
| q+h | 92.6540 | 92.0000 | 119 | 0 |

## Retrieval Sensitivity

| Comparator | Target | n | Target any-hit | Target all-hit | Target recall | Delta all-hit [95% CI] | Delta recall [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| Single rewritten query | Length-matched hypothetical only | 500 | 0.9200 | 0.2740 | 0.5970 | -0.3440 [-0.3940, -0.2980] | -0.2020 [-0.2310, -0.1730] |
| Question + rewritten query | Length-matched question + hypothetical | 500 | 0.9740 | 0.6400 | 0.8070 | 0.0120 [-0.0240, 0.0480] | 0.0040 [-0.0170, 0.0250] |
