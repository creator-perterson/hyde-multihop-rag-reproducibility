from __future__ import annotations

import bz2
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_DOCUMENT_COUNT = 5_233_329


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_list_sha256(ids: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def row_id(row: dict) -> str:
    for key in ("question_id", "id", "_id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError("question row has no question_id, id, or _id")


def align_hotpot_ids(current_rows: list[dict], official_rows: list[dict]) -> list[dict]:
    requested_ids = [row_id(row) for row in current_rows]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("duplicate IDs in requested evaluation rows")

    official_by_id: dict[str, dict] = {}
    for row in official_rows:
        identifier = row_id(row)
        if identifier in official_by_id:
            raise ValueError(f"duplicate official ID: {identifier}")
        official_by_id[identifier] = row

    missing = [identifier for identifier in requested_ids if identifier not in official_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"missing {len(missing)} requested IDs from official rows: {preview}")
    return [official_by_id[identifier] for identifier in requested_ids]


def count_bz2_jsonl_documents(corpus_root: Path) -> tuple[int, int]:
    files = sorted(corpus_root.glob("*/wiki_*.bz2"))
    if not files:
        raise ValueError(f"no wiki_*.bz2 shards found under {corpus_root}")
    count = 0
    for path in files:
        with bz2.open(path, "rb") as source:
            count += sum(1 for line in source if line.strip())
    return count, len(files)


def build_corpus_manifest(
    archive_path: Path,
    corpus_root: Path,
    question_rows: list[dict],
    document_count: int,
    shard_count: int,
) -> dict:
    question_ids = [row_id(row) for row in question_rows]
    manifest = {
        "schema_version": 1,
        "dataset": "hotpotqa",
        "corpus_scope": "full-wiki",
        "corpus_variant": "introductory-paragraph abstracts used by official IRCoT",
        "wikipedia_snapshot": "2017-10-01",
        "source_urls": [
            "https://nlp.stanford.edu/projects/hotpotqa/"
            "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
        ],
        "upstream_repository": "https://github.com/StonyBrookNLP/ircot",
        "upstream_commit": "3c1820f698eea5eeddb4fba3c56b64c961e063e4",
        "license_provenance": "HotpotQA and the 2017-10-01 English Wikipedia snapshot",
        "document_count": document_count,
        "expected_document_count": EXPECTED_DOCUMENT_COUNT,
        "shard_count": shard_count,
        "files": [
            {
                "logical_path": "downloads/"
                "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
                "bytes": archive_path.stat().st_size,
                "md5": file_digest(archive_path, "md5"),
                "sha256": file_digest(archive_path, "sha256"),
            }
        ],
        "processed_layout": {
            "logical_root": "raw/hotpotqa/wikpedia-paragraphs",
            "glob": "*/wiki_*.bz2",
        },
        "evaluation": {
            "split": "official test_subsampled IDs",
            "question_count": len(question_ids),
            "question_ids_sha256": id_list_sha256(question_ids),
        },
    }
    validate_fullwiki_manifest(manifest)
    return manifest


def validate_fullwiki_manifest(manifest: dict) -> None:
    if manifest.get("dataset") != "hotpotqa":
        raise ValueError("dataset must be hotpotqa")
    if manifest.get("corpus_scope") != "full-wiki":
        raise ValueError("corpus_scope must be full-wiki")
    document_count = manifest.get("document_count")
    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError("document_count must be a positive integer")
    expected = manifest.get("expected_document_count")
    if expected is not None and document_count != expected:
        raise ValueError(
            f"document_count {document_count} does not match expected {expected}"
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("files must contain at least one manifest entry")
    for entry in files:
        sha256 = str(entry.get("sha256", "")).lower()
        if not SHA256_RE.fullmatch(sha256):
            raise ValueError("every file requires a lowercase SHA-256 digest")
        logical_path = str(entry.get("logical_path", ""))
        if not logical_path or Path(logical_path).is_absolute():
            raise ValueError("file paths must be non-empty logical paths")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0:
            raise ValueError("every file requires a positive byte count")

    evaluation = manifest.get("evaluation", {})
    if evaluation.get("question_count") != 500:
        raise ValueError("evaluation must contain exactly 500 questions")
    if not SHA256_RE.fullmatch(str(evaluation.get("question_ids_sha256", ""))):
        raise ValueError("evaluation requires an ordered question ID SHA-256")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
