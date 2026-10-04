# Third-Party Notices

This repository is an artifact-audit and table-reconstruction package for the accompanying manuscript. It includes project code, paper sources, compiled PDFs, lightweight derived summaries, hash/provenance tables, and scripts for audit and table reconstruction. It does not relicense third-party datasets, processed corpora, model checkpoints, hosted-model services, or unrestricted raw model-output dumps.

## Project-authored materials

- Project-authored software code and associated software documentation are licensed under the MIT License in `LICENSE`.
- Paper text, manuscript PDFs, and figure assets are not automatically covered by the MIT software license unless a file explicitly says otherwise. They are provided for scholarly review, audit, and manuscript inspection. Any later publisher or venue license may supersede this local review-package statement for the published version.
- Lightweight CSV/MD summaries and LaTeX table snapshots are provided to support audit and reconstruction of the reported aggregate tables. They should not be treated as a redistribution of the full underlying datasets or full hosted-model outputs.

## Datasets and processed corpora

The manuscript uses released IRCoT processed HotpotQA and 2WikiMultihopQA 500-example test splits under a processed local-corpus protocol. Full processed datasets, generated local corpora, FAISS indexes, embedding caches, and large derived caches are intentionally excluded from this repository.

- HotpotQA data and code remain governed by the upstream HotpotQA terms and licenses. This repository does not relicense HotpotQA data.
- 2WikiMultihopQA data and code remain governed by the upstream 2WikiMultihopQA terms and licenses. This repository does not relicense 2WikiMultihopQA data.
- IRCoT processed artifacts, scripts, and mirrors remain governed by their upstream terms and any terms attached to the source from which a user obtains them. This repository does not relicense IRCoT processed data.
- Users are responsible for obtaining any required upstream datasets or processed artifacts from authorized sources and for following the applicable upstream licenses and terms.

The confirmatory Query2doc adaptation derives four-shot demonstration passages
from authorized local copies of the released IRCoT HotpotQA and
2WikiMultihopQA training examples. Demonstration text and the local
demonstration pool are excluded from the public artifact release; only
demonstration IDs and prompt hashes are retained for provenance. Query2doc is
credited to Wang et al., *Query2doc: Query Expansion with Large Language Models*
(EMNLP 2023), and this repository does not relicense its paper, code, or data.

The local data policy is manifest-and-script based: this repository provides provenance records, hashes, preparation scripts, and lightweight derived summaries so that authorized local copies can be checked against the reported artifacts. See `docs/data_provenance.md` and `artifacts/hashes/` for the recorded source and hash information.

## Official IRCoT full-wiki confirmatory tier

The full-wiki confirmatory tier uses reference code from the official
`StonyBrookNLP/ircot` repository at commit
`3c1820f698eea5eeddb4fba3c56b64c961e063e4`, distributed upstream under the
Apache License 2.0. The upstream URL, archive hash, commit, license hash, and
selected source-file hashes are recorded in
`artifacts/summaries/ircot_fullwiki/upstream_manifest.json`. The local Qwen
adapter changes only the model-serving boundary; it does not relicense or claim
authorship of the official IRCoT algorithm, prompts, or evaluation logic.

HotpotQA full-wiki paragraphs and their Wikipedia-derived contents remain
subject to the upstream HotpotQA and Wikipedia licenses, including the
applicable CC BY-SA terms. Raw archives, processed paragraph files, and local
Elasticsearch indexes are not redistributed. Elasticsearch and its bundled or
separate Java runtime remain governed by their upstream licenses. Users must
obtain these dependencies and data from authorized upstream sources.

## Hosted-model APIs and generated outputs

The manuscript reports hosted Qwen-family model calls through the Alibaba Cloud Bailian/DashScope OpenAI-compatible chat-completions interface.

- API credentials, private endpoint URLs, request identifiers, billing exports, latency logs, and provider raw request/response envelopes are not included.
- Qwen API access and outputs are subject to the applicable provider terms. This repository does not grant rights to use the provider service or to redistribute unrestricted raw hosted-model output dumps.
- Included lightweight summaries, aggregate statistics, prompt hashes, model-string records, and table-reconstruction artifacts are intended for audit of the reported results, not as a provider-side request log or an exact replay guarantee for hosted aliases.

## Model assets and software dependencies

Dense retrieval experiments refer to third-party model assets such as `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`. These model assets are not relicensed by this repository. Users should obtain them through their normal upstream distribution channels and follow the corresponding model-card licenses and usage terms.

The open-weight confirmatory tier uses `Qwen/Qwen2.5-7B-Instruct` at revision `fe11104b620d588ccc049ff6631dd3ea002e3d98`. The locally cached model card and checkpoint license file identify Apache-2.0. The checkpoint is not redistributed by this repository; users must obtain it from an authorized upstream source and comply with its license and model-card terms. Included generated-text artifacts remain subject to applicable dataset, model, and downstream-use obligations.

Python dependencies are listed in `requirements.txt`, with a stricter local environment snapshot in `requirements-lock.txt`. Third-party Python packages remain governed by their own licenses.

## Springer Nature LaTeX template

The JIIS submission source includes `sn-jnl.cls` and `sn-basic.bst` from the
Springer Nature journal article template identified on the official LaTeX author
support page as the December 2024 version. The template remains governed by its
upstream terms and is included only to compile and submit the manuscript. The
official source page, download URL, retrieval date, and file hashes are recorded
in `paper/jiis/template_manifest.json`. The user supplied an extracted official
template directory rather than the original downloaded ZIP. Consequently, the
recorded archive hash belongs to a local preservation ZIP rebuilt from that
directory and is explicitly not claimed to be the server-original ZIP hash.

## Follow-up resource and whitelist register (checked 2026-10-04)

The local capsule version `0.1.0-local-20261004` contains only the files enumerated with hashes in its `manifest.json`. Public metric exports contain example IDs, method identifiers and numeric metrics; retrieval titles, paragraph text, prompts and generated text have been removed. `export_provenance.json` binds the exports to existing sanitized open-generator records. Frozen aggregate CSV files support table audit only; they do not grant rights in the original data.

| Resource | Official source / observed terms | Capsule treatment and acquisition |
|---|---|---|
| HotpotQA and processed Wikipedia | [HotpotQA homepage](https://hotpotqa.github.io/): CC BY-SA 4.0 | No question, answer or Wikipedia text; acquire upstream data separately and retain attribution/share-alike obligations. |
| 2WikiMultihopQA | [Upstream LICENSE](https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE): Apache 2.0 repository notice | No corpus text; obtain official release and check accompanying data provenance and third-party source terms. Repository software license is not blanket clearance for all Wikipedia-derived material. |
| IRCoT reference software | [Upstream LICENSE](https://github.com/StonyBrookNLP/ircot/blob/main/LICENSE): Apache 2.0 | Not vendored in this minimal capsule; full run requires pinned official code and authorized processed data. |
| MultiHop-RAG news corpus | Upstream repository/model-card retrieval did not succeed during this check; precise corpus/news reuse scope remains unverified | No news text or raw predictions included; upstream data and publisher rights must be checked before any future redistribution. |
| Qwen2.5-7B-Instruct | [Official model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct): Apache 2.0 | No weights or generated text; acquire pinned revision listed in manuscript/provenance. |
| Mistral-7B-Instruct-v0.3 | [Official model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3): Apache 2.0 | No weights; full runs require the recorded revision and tokenizer. |
| BGE base en v1.5 | [Official model card](https://huggingface.co/BAAI/bge-base-en-v1.5): MIT | No weights; acquire pinned checkpoint separately. |
| all-MiniLM-L6-v2 | [Official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): Apache 2.0 | No weights included. |
| cross-encoder/ms-marco-MiniLM-L6-v2 | [Official model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): Apache 2.0 | No weights included; training-data rights remain separate. |
| Hosted Qwen/Bailian calls | Provider contract and applicable account/service terms; no blanket output redistribution license verified here | No credentials, endpoint, raw request/response, answer or generated text; acquire service access separately. Historical aliases are not immutable checkpoints. |
| Springer template and paper assets | Existing template provenance/terms and scholarly review scope above | Not included in minimal statistical capsule; manuscript-source compilation is a separate layer. |
| Python/packages | Capsule uses Python 3.10+ standard library; other experiment dependencies remain independently licensed | No Python distribution or external libraries bundled; existing requirements snapshots describe full experiments, not capsule prerequisites. |

MIT applies to project-authored software, including the capsule wrapper, and does not replace these third-party conditions. This register records official notices, not a legal determination of unrestricted reuse. Existing generated-text files and text-containing retrieval files are excluded. No upload or DOI deposition was performed.
