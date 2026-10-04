# JIIS numerical reproduction guide

Current entry: `reproducibility/README.md`. The two frozen packages were prepared on 2026-10-04 and retain their original versions/manifests. Their subsequent distribution is recorded separately in `release/github_publication_manifest.json`; no archival DOI is assigned.

| Reproduction layer | Supplied scope | Limits |
|---|---|---|
| Manuscript compilation | `paper/jiis/` and source ZIPs in `release/assets/` | Requires a compatible LaTeX toolchain; this is not an experimental run. |
| Table reconstruction | `reproducibility/legacy/` rebuilds legacy and two open-generator tables | Current JIIS tables are supplied as manuscript sources; not every private-source builder is included. |
| Paired numerical reanalysis | Legacy: eight historical retrieval comparisons. Follow-up: four serializer and 27 random 300-question comparisons, 615 numeric fields | Exact supplied vector-based scope only; not all historical news/reader families. |
| Raw-prediction rescoring | Not provided | Excluded raw predictions/gold text and missing historical hosted answer dumps prevent this claim. |
| Full model runs | Not provided by the capsules | Requires upstream data, model/tokenizer revisions, indexes, effective inputs, adapters and runtime. Missing external IRCoT, incomplete costs and absent independent human validation remain limits. |

## Legacy commands

From `reproducibility/legacy/`, run `python -S run.py` with Python 3.10+. No third-party dependencies, credentials, GPU, weights or search service are required. The wrapper verifies its manifest and checks eight paired comparisons (n=500 per dataset), frozen Holm family8 and two byte-identical open-generator tables. Legacy table reconstruction is a separate output.

## Follow-up commands

From `reproducibility/followup/`, install `requirements.txt` (NumPy 2.2.6), then run:

```powershell
python -I scripts/reanalyze_jiis_followup_metrics.py --input-root data --out outputs
python -I -m unittest discover -s tests -p test_followup_metric_reanalysis.py -v
```

Four serializer contrasts preserve EM family2 and F1 family2. The 27 holdout contrasts preserve EM6, retrieval3, secondary F1 6, official-EM sensitivity6 and official-F1 sensitivity6. Paired directions, ordered identifiers and family sizes are retained. Available score means and sanitized support counts are checked against summaries at absolute tolerance1e-12. Runtime, tokenization, effective-text and semantic-sufficiency verification are outside this capsule. Input-rejection tests cover missing/duplicate identifiers, changed pairing/summary order, changed frozen p values, private fields and nonfinite metrics.

## Public boundary

The manifests enumerate approved package files; a folder name containing `public` is not permission to publish everything in it. Supplied inputs contain numeric metrics, method/reader identifiers and ID/hash anchors. They omit raw benchmark/corpus/news text, gold answers, generated answers, prompts, reasoning traces, private endpoints/credentials and model/index binaries. Text-containing local news retrieval files and older local submission bundles are excluded. Project software is MIT; upstream resources and NumPy retain their own terms.

## Packaging and publication provenance

The original ZIPs retain their recorded hashes and packaging-time metadata. Copying the readable files to `reproducibility/legacy/` and `reproducibility/followup/` does not change their statistical inputs or version identity. Use the root publication manifest for the distribution record. Package verification does not establish full scientific model reproduction.

## Current article and supplementary-material synchronization

**Reliability Boundaries of HyDE-Style Query Expansion in Multi-Hop Intelligent Information Retrieval** is the current article title in the manuscript, Supplementary Information, source archives and ESM wrapper identification. The final abstract distinguishes generators from readers and the original hosted artifact from separate matched BGE-encoder comparisons. The Introduction and Conclusion use matched question-only baselines and limit claims to tested configurations.

| Material | Current public file | Synchronization and scope |
|---|---|---|
| Online Resource 1 | [Supplementary Information](../paper/jiis/supplementary_jiis.pdf) | Current title and identification; 69 pages, six wide-table pages displayed in landscape. |
| Online Resource 2 | [Article-identified ZIP](../release/assets/jiis_ESM_2_submission_20261005.zip) | Current article metadata and usage instructions around the unchanged legacy numerical capsule. |
| Online Resource 3 | [Article-identified ZIP](../release/assets/jiis_ESM_3_submission_20261005.zip) | Current article metadata and usage instructions around the unchanged follow-up numerical capsule. |

Both ZIP wrappers use packaging revision `jiis-materials-sync-20261005` in their README and JSON metadata. Original frozen capsule versions, manifests, inputs and archive bytes remain unchanged. Current wrapper and manuscript/source hashes are recorded in [the publication manifest](../release/github_publication_manifest.json). Local submission cover letters and internal revision records are not part of the public package.
