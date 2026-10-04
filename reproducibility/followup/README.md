# Public F1 metric reanalysis capsule

Version0.2.0-local-20261004; local only, no DOI or upload. The older0.1.0 legacy capsule is unchanged.

Python3.10+ and NumPy2.2.6. Install `python -m pip install -r requirements.txt` in your own environment, then run from this directory:

```
python -I scripts/reanalyze_jiis_followup_metrics.py --input-root data --out outputs
python -I -m unittest discover -s tests -p test_followup_metric_reanalysis.py -v
```

Expected:4 serializer and27 holdout paired contrasts match, including frozen n/order/family/p values and available adapted/raw score means at absolute tolerance1e-12. All source/input hashes are in manifest.json/source_provenance.json. Tampered/reordered/duplicate/missing/nonfinite public vectors are rejected. Existing infer, exact McNemar and Holm algorithms are imported unchanged; this package never reads predictions or gold, calls a scorer, runs a tokenizer/model, or contacts a search service.

Serializer EOS/cap/timing/token summary verification stays with the private independent408-row validator and is excluded from this score-vector reanalysis. If raw official holdout means are not exported, their exclusion is recorded in verification.json. Strict support counts can be reconstructed from sanitized booleans but are not semantic sufficiency. No new endpoint or statistical family is introduced. This establishes numeric-vector reanalysis only, not raw-prediction rescoring or scientific model reproduction.

Project-authored software is MIT as in LICENSE. NumPy remains under its upstream BSD license (https://numpy.org/doc/stable/license.html); no distribution is bundled. The public numeric exports contain method/reader IDs, question ID/hash anchors and scores only. Upstream datasets/models/corpora are excluded and remain independently licensed. Refer to the repository THIRD_PARTY_NOTICES.md and docs/jiis_followup_reproduction_guide.md for acquisition and broader limits.
