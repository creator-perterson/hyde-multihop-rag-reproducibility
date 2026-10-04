# Reproducing the numerical results

Run each package from its own directory. The package copies are byte-identical to their frozen downloadable archives; do not edit files listed in their manifests.

## Legacy package: 0.1.0-local-20261004

Python 3.10+ standard library only:

```powershell
cd reproducibility/legacy
python -S run.py
```

Expected final message: `PASS: eight paired contrasts match; two open-generator tables byte-identical; legacy table rebuilt.` Results are written under `outputs/` and are not source inputs.

## Follow-up package: 0.2.0-local-20261004

Python 3.10+ and NumPy 2.2.6:

```powershell
cd reproducibility/followup
python -m pip install -r requirements.txt
python -I scripts/reanalyze_jiis_followup_metrics.py --input-root data --out outputs
python -I -m unittest discover -s tests -p test_followup_metric_reanalysis.py -v
```

Expected: 31 paired contrasts and 615 numeric fields agree with the supplied frozen summaries; seven input-rejection tests pass. Outputs include `outputs/verification.json`. These commands use numerical score/support vectors and neither read raw predictions/gold text nor call a model, tokenizer, scorer or search service.

## Downloads and historical metadata

Corresponding original ZIPs are in `../release/assets/`. Capsule manifests and documentation describe their state when packaged on 2026-10-04. Their historical `local_only` fields are not rewritten on publication; `../release/github_publication_manifest.json` records the later distribution and hashes. Versions remain `0.1.0-local-20261004` and `0.2.0-local-20261004`. No archival DOI is assigned.

## Scope and missing inputs

Numerical reanalysis is distinct from raw-answer rescoring and model reproduction. The two packages omit raw questions, gold answers, generated answers, corpus/news text, prompts, model weights and indexes. Token/timing/serializer diagnostics outside the supplied vectors require additional original inputs. Independent human validation was not performed. See `../docs/jiis_followup_reproduction_guide.md` and `../THIRD_PARTY_NOTICES.md` for result-specific limits and separate upstream terms.
