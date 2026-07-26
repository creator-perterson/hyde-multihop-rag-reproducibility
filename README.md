# HyDE-Style Query Expansion for Multi-Hop Evidence Acquisition

This repository contains the code, paper sources, final manuscript PDFs, and lightweight reproducibility artifacts for:

**A Controlled Study of HyDE-Style Query Expansion for Multi-Hop Evidence Acquisition**

The anonymized review repository is available at:

https://github.com/creator-perterson/hyde-multihop-rag-reproducibility

The repository is prepared for GitHub release without local secrets, virtual environments, build caches, raw processed datasets, private endpoint URLs, or large embedding/retrieval caches.

## Repository Layout

```text
configs/                 Optional experiment configuration files.
experiments/             PowerShell runners for equal-budget diagnostics.
src/                     Data preparation, retrieval, generation, evaluation, and verifier code.
tests/                   Unit and regression tests used during manuscript preparation.
paper/
  manuscript_v0.pdf      Current compiled main manuscript.
  supplemental_material.pdf
  latex/                 LaTeX source files and bibliography.
  figures/               Final exported figure assets.
artifacts/
  summaries/             Lightweight CSV/MD summaries used by the paper tables.
  hashes/                LaTeX provenance/hash tables from the supplement.
docs/                    Reproducibility notes, model invocation ledger, and upload steps.
```

## What Is Not Included

- Real API keys or `.env` files.
- Full processed datasets and generated local corpora.
- FAISS indexes, embedding caches, and `.pt` cache files.
- Full hosted-reader output dumps unless represented by lightweight summaries.
- LaTeX temporary files and visual-check screenshots.

The paper reports upstream dataset commits, file hashes, and conversion outputs in the supplement and in `artifacts/hashes/`. When upstream dataset licenses or terms restrict redistribution, this repository provides provenance, manifests, hashes, preparation scripts, and derived table summaries rather than republishing the full processed corpora.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For hosted LLM calls, copy `.env.example` to `.env` and fill in local credentials. Do not commit `.env`.

## Quick Checks

```powershell
pytest tests
```

Some tests or full reproduction scripts require local datasets, model caches, or hosted LLM access. The included summaries and manuscript PDFs are intended to support audit and review without uploading large generated artifacts.

## Rebuild Main Tables From Frozen Summaries

This command rebuilds compact LaTeX snapshots of the main reported tables from the included frozen CSV/MD summaries, without API keys or raw datasets:

```powershell
python scripts/rebuild_main_tables_from_summaries.py
```

The output is written to `artifacts/tables/rebuilt_main_tables.tex`.

## GitHub Upload

See `docs/upload_steps.md`.
