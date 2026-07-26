# Data Provenance Notes

This repository does not redistribute the full processed HotpotQA or 2WikiMultihopQA corpora.

License check for the upstream datasets:

- HotpotQA distributes its dataset under CC BY-SA 4.0 and its code under Apache-2.0.
- The official 2WikiMultihopQA release is distributed under Apache-2.0.

Because redistribution terms can differ between upstream raw datasets, processed mirrors, and derived local corpora, this review package keeps the full corpora out of Git and publishes only manifests, hashes, preparation scripts, and frozen lightweight summaries.

The manuscript uses released IRCoT processed 500-example test splits and records upstream source, commit, raw file names, raw SHA-256 prefixes, and converted artifact SHA-256 prefixes in:

- `artifacts/hashes/table_dataset_artifact_provenance.tex`
- `paper/latex/table_dataset_artifact_provenance.tex`
- `paper/supplemental_material.pdf`

The intended workflow is:

1. Obtain the released IRCoT processed artifacts from the upstream source recorded in the supplement.
2. Verify the raw file hashes against the provenance table.
3. Run the conversion/preparation scripts under `src/data/`.
4. Verify converted `questions.jsonl` and `corpus.jsonl` hashes against the supplement.

Large derived corpora, FAISS indexes, embedding caches, and hosted-reader answer dumps are intentionally excluded from GitHub. Lightweight CSV/MD summaries used by the paper tables are stored under `artifacts/summaries/`.

The review release therefore follows a manifest-and-script policy for data that cannot be directly redistributed: reviewers should obtain authorized upstream copies, verify hashes, run the preparation scripts, and compare regenerated summaries against the included frozen summaries. The table-rebuild entry point is:

```powershell
python scripts/rebuild_main_tables_from_summaries.py
```
