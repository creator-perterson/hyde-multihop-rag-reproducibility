# HyDE-Style Query Expansion: Reliability Boundaries

Code, manuscript sources and numerical audit materials for:

**When Does HyDE-Style Query Expansion Help? Reliability Boundaries in Multi-Hop Intelligent Information Retrieval**

Shiyong Xiong and Xingyun Chen, School of Computer Science and Technology, Chongqing University of Posts and Telecommunications, Chongqing, China.

## Current manuscript

- [Main manuscript](paper/jiis/manuscript_jiis.pdf): 25 pages.
- [Supplementary Information](paper/jiis/supplementary_jiis.pdf): 69 pages, including the additional serializer and random 300-question experiments.
- [Editable LaTeX sources](paper/jiis/): bibliography, figures and Springer class/style included.
- [Versioned downloads](release/assets/): two numeric capsules and manuscript-source archives.

The JIIS manuscript is the current version. Earlier PDFs and `paper/latex/` files retained from the initial repository commit are historical drafts.

## Numerical reproduction

See [the reproduction entry point](reproducibility/README.md) and [scope guide](docs/jiis_followup_reproduction_guide.md).

| Package | Command from its directory | Scope |
|---|---|---|
| [Legacy capsule](reproducibility/legacy/) | `python -S run.py` | Legacy table reconstruction and eight historical paired retrieval comparisons; Python 3.10+ standard library |
| [Follow-up capsule](reproducibility/followup/) | `python -I scripts/reanalyze_jiis_followup_metrics.py --input-root data --out outputs` | Four serializer and 27 new-question comparisons, 615 numeric fields; Python 3.10+ and NumPy 2.2.6 |

For the follow-up capsule, first install its `requirements.txt` in your environment. It also includes seven input-rejection tests.

These packages reproduce specified numerical results from frozen vectors. They do not rescore raw predictions, rerun models, rebuild full corpora/indexes or reproduce every result in the article. Corpora, questions, gold answers, raw predictions, prompts, reasoning traces, credentials, private request logs, model weights and indexes are excluded from this update. Existing code from the initial commit remains available, but the current complete model-running environment is not supplied by these capsules.

## Provenance and licensing

[Publication manifest](release/github_publication_manifest.json) binds the current numerical packages and paper files to their SHA-256 hashes. Frozen capsule manifests retain their original packaging-time `local_only` status and versions; the publication manifest records subsequent distribution separately. No archival DOI has been assigned.

Project-authored software and associated software documentation are under [MIT](LICENSE). Paper text, manuscript PDFs, figures, third-party datasets/models and the Springer template retain their separate terms; see [Third-party notices](THIRD_PARTY_NOTICES.md).

## Publication maintenance

Use an explicit file allowlist and inspect staged changes. [Maintenance instructions](docs/upload_steps.md) explain the publication boundaries. Internal submission files, historical delivery directories, temporary files and raw experimental data are not part of this update.
