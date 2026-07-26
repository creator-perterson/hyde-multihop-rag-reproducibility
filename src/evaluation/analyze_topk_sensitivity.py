import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_KS = (5, 10, 20)


# RRF fusion is sensitive to the per-branch candidate depth. The top-20 Hybrid
# artifacts therefore do not reproduce the canonical top-10 Hybrid rows.
DEPTH_SENSITIVE_FUSION_METHODS = {"Hybrid"}


DEFAULT_CONFIGS = [
    {
        "dataset": "HotpotQA",
        "method": "Dense",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_hotpotqa_test500_dense_top20_retrieval.jsonl",
    },
    {
        "dataset": "HotpotQA",
        "method": "BM25",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_hotpotqa_test500_bm25_top20_retrieval.jsonl",
    },
    {
        "dataset": "HotpotQA",
        "method": "Direct rewrite",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_hotpotqa_test500_direct_rewrite_top20_retrieval.jsonl",
    },
    {
        "dataset": "HotpotQA",
        "method": "HyDE",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_hotpotqa_test500_hyde_top20_retrieval.jsonl",
    },
    {
        "dataset": "2WikiMultihopQA",
        "method": "Dense",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_2wiki_test500_dense_top20_retrieval.jsonl",
    },
    {
        "dataset": "2WikiMultihopQA",
        "method": "BM25",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_2wiki_test500_bm25_top20_retrieval.jsonl",
    },
    {
        "dataset": "2WikiMultihopQA",
        "method": "Direct rewrite",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_2wiki_test500_direct_rewrite_top20_retrieval.jsonl",
    },
    {
        "dataset": "2WikiMultihopQA",
        "method": "HyDE",
        "path": "local-artifacts/topk_sensitivity/retrieval/ircot_2wiki_test500_hyde_top20_retrieval.jsonl",
    },
]


def normalize_title(text):
    return " ".join(str(text).lower().strip().split())


def support_titles(row):
    facts = row.get("supporting_facts", {})
    titles = []
    if isinstance(facts, dict):
        raw_titles = facts.get("title", [])
        titles.extend(raw_titles if isinstance(raw_titles, list) else [raw_titles])
    elif isinstance(facts, list):
        for item in facts:
            if isinstance(item, dict):
                titles.append(item.get("title", ""))
            elif isinstance(item, (list, tuple)) and item:
                titles.append(item[0])
            else:
                titles.append(item)

    normalized = []
    seen = set()
    for title in titles:
        key = normalize_title(title)
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    return normalized


def first_support_ranks(row):
    wanted = set(support_titles(row))
    ranks = {}
    for rank, doc in enumerate(row.get("retrieved", []), start=1):
        key = normalize_title(doc.get("title", ""))
        if key in wanted and key not in ranks:
            ranks[key] = rank
    return ranks


def summarize_rows(rows, ks=DEFAULT_KS, rank_depth=None):
    rows = list(rows)
    if rank_depth is None:
        rank_depth = max((len(row.get("retrieved", [])) for row in rows), default=0)
    capped_missing_rank = rank_depth + 1

    all_hits = {k: [] for k in ks}
    recalls = {k: [] for k in ks}
    worst_ranks = []
    missing_worst = 0

    for row in rows:
        titles = support_titles(row)
        ranks = first_support_ranks(row)
        capped_ranks = [ranks.get(title, capped_missing_rank) for title in titles]
        if any(rank > rank_depth for rank in capped_ranks):
            missing_worst += 1
        if capped_ranks:
            worst_ranks.append(max(capped_ranks))

        for k in ks:
            if not titles:
                all_hits[k].append(0.0)
                recalls[k].append(0.0)
                continue
            hits = sum(1 for title in titles if ranks.get(title, capped_missing_rank) <= k)
            recalls[k].append(hits / len(titles))
            all_hits[k].append(1.0 if hits == len(titles) else 0.0)

    summary = {
        "n": len(rows),
        "rank_depth": rank_depth,
        "missing_worst_rank_count": missing_worst,
        "mean_worst_support_rank_capped": statistics.mean(worst_ranks) if worst_ranks else 0.0,
        "median_worst_support_rank_capped": statistics.median(worst_ranks) if worst_ranks else 0.0,
    }
    for k in ks:
        summary[f"all_hit_at{k}"] = statistics.mean(all_hits[k]) if all_hits[k] else 0.0
        summary[f"recall_at{k}"] = statistics.mean(recalls[k]) if recalls[k] else 0.0
    return summary


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path, rows, ks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "n",
        "rank_depth",
        *[f"all_hit_at{k}" for k in ks],
        *[f"recall_at{k}" for k in ks],
        "mean_worst_support_rank_capped",
        "median_worst_support_rank_capped",
        "missing_worst_rank_count",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value, digits=4):
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_markdown(path, rows, ks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Dataset",
        "Method",
        *[f"All@{k}" for k in ks],
        *[f"Recall@{k}" for k in ks],
        "Capped mean worst rank",
        "Capped median worst rank",
        "Miss@20",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["dataset"],
            row["method"],
            *[fmt(row[f"all_hit_at{k}"]) for k in ks],
            *[fmt(row[f"recall_at{k}"]) for k in ks],
            fmt(row["mean_worst_support_rank_capped"], 2),
            fmt(row["median_worst_support_rank_capped"], 1),
            str(row["missing_worst_rank_count"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows, ks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if row["method"] not in DEPTH_SENSITIVE_FUSION_METHODS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Top-$k$ sensitivity and supporting-title rank audit for fixed-ranking retrieval variants.}",
        r"\label{tab:topk_sensitivity}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{llrrrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & All@5 & All@10 & All@20 & Rec@5 & Rec@10 & Rec@20 & Capped mean worst rank & Capped median worst rank & Miss@20 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["dataset"],
                    row["method"],
                    fmt(row["all_hit_at5"]),
                    fmt(row["all_hit_at10"]),
                    fmt(row["all_hit_at20"]),
                    fmt(row["recall_at5"]),
                    fmt(row["recall_at10"]),
                    fmt(row["recall_at20"]),
                    fmt(row["mean_worst_support_rank_capped"], 2),
                    fmt(row["median_worst_support_rank_capped"], 1),
                    str(row["missing_worst_rank_count"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item All rows are rescored from top-20 retrieval artifacts generated from frozen questions and frozen generated query text; no new LLM calls are used. The depth-sensitive RRF fusion control is omitted because changing the per-branch candidate depth for top-20 fusion changes the fused top-10 ranking relative to the canonical fusion rows in Tables~\ref{tab:main_results} and~\ref{tab:2wiki_generalization}. All@k requires all annotated supporting titles to appear within the first k retrieved documents. Rec@k is mean supporting-title recall at k. Capped worst-rank columns use the worst supporting-title rank per question, with supports not retrieved in the top 20 capped at 21; Miss@20 counts examples with at least one supporting title still absent by rank 20.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_configs(paper_root, configs, ks=DEFAULT_KS, rank_depth=20):
    rows = []
    for config in configs:
        path = Path(config["path"])
        if not path.is_absolute():
            path = Path(paper_root) / path
        retrieval_rows = list(read_jsonl(path))
        summary = summarize_rows(retrieval_rows, ks=ks, rank_depth=rank_depth)
        rows.append(
            {
                "dataset": config["dataset"],
                "method": config["method"],
                "source_file": str(path),
                **summary,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--rank_depth", type=int, default=20)
    parser.add_argument(
        "--out_dir",
        default="local-artifacts/topk_sensitivity",
    )
    parser.add_argument(
        "--latex_out",
        default="paper/latex/table_topk_sensitivity.tex",
    )
    args = parser.parse_args()

    paper_root = Path(args.paper_root)
    out_dir = paper_root / args.out_dir
    rows = analyze_configs(paper_root, DEFAULT_CONFIGS, ks=DEFAULT_KS, rank_depth=args.rank_depth)
    write_csv(out_dir / "topk_sensitivity_summary.csv", rows, DEFAULT_KS)
    write_markdown(out_dir / "topk_sensitivity_summary.md", rows, DEFAULT_KS)
    write_latex(paper_root / args.latex_out, rows, DEFAULT_KS)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
