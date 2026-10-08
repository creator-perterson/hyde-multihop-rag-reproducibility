# HyDE-Style Query Expansion: Reliability Boundaries

Code, manuscript sources and numerical audit materials for:

**Reliability Boundaries of HyDE-Style Query Expansion in Multi-Hop Intelligent Information Retrieval**

Xingyun Chen and Shiyong Xiong, School of Computer Science and Technology, Chongqing University of Posts and Telecommunications, Chongqing, China.

Corresponding author: Xingyun Chen (cxy201318@qq.com).

## Current manuscript

- [Main manuscript](paper/jiis/manuscript_jiis.pdf): 25 pages.
- [Supplementary Information](paper/jiis/supplementary_jiis.pdf): 69 pages, including the additional serializer and random 300-question experiments.
- [Editable LaTeX sources](paper/jiis/): bibliography, figures and Springer class/style included.
- [Main source archive](release/assets/jiis_main_source_20261008_keywords.zip) and [supplement source archive](release/assets/jiis_supplement_source_20261008.zip): synchronized with the current PDFs.
- [Online Resource 2](release/assets/jiis_ESM_2_submission_20261008.zip) and [Online Resource 3](release/assets/jiis_ESM_3_submission_20261008.zip): numerical capsules with article title, authors, affiliation, corresponding contact and usage instructions.
- [Original frozen downloads](release/assets/): the two numerical capsule archives retain their original bytes and versions.

The JIIS manuscript is the current version. The 2026-10-08 front-matter layout revision places the complete abstract and all five keywords together on page one, with unchanged text, font sizes and page margins. Earlier PDFs and `paper/latex/` files retained from the initial repository commit are historical drafts. The 2026-10-05 revision adopts the reliability-boundaries title and the finalized abstract, limits Introduction/Conclusion claims to tested configurations, uses consistent matched question-only baseline terminology, and displays the six wide supplementary tables in landscape orientation. The ESM wrappers add article identification outside the unchanged frozen capsules; their ZIP hashes differ from the original capsule archives.

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
