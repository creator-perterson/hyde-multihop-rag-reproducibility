import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DATASETS = {
    "hotpotqa": {
        "label": "HotpotQA",
        "dataset_dir": "local-artifacts/datasets/ircot_hotpotqa_test500",
        "main_retrievals": [
            ("Main", "Dense", "local-artifacts/results/ircot_hotpotqa_test500_top10_retrieval.jsonl"),
            (
                "Main",
                "Multi-query",
                "local-artifacts/results/ircot_hotpotqa_test500_multiquery_top10_decay050_retrieval.jsonl",
            ),
            ("Main", "BM25", "local-artifacts/results/ircot_hotpotqa_test500_bm25_top10_retrieval.jsonl"),
            ("Main", "Hybrid", "local-artifacts/results/ircot_hotpotqa_test500_hybrid_top10_retrieval.jsonl"),
            (
                "Main",
                "Rule-based iterative",
                "local-artifacts/results/ircot_hotpotqa_test500_iterative_top10_retrieval.jsonl",
            ),
            (
                "Main",
                "Direct rewrite",
                "local-artifacts/results/ircot_hotpotqa_test500_single_query_reformulation_top10_retrieval.jsonl",
            ),
            (
                "Main",
                "Question + rewrite",
                "local-artifacts/results/ircot_hotpotqa_test500_question_plus_single_query_reformulation_top10_retrieval.jsonl",
            ),
            (
                "Main",
                "Hypothetical-only",
                "local-artifacts/results/ircot_hotpotqa_test500_hyde_hypothetical_only_top10_retrieval.jsonl",
            ),
            ("Main", "HyDE", "local-artifacts/results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl"),
        ],
        "equal_budget_prefix": "ircot_hotpotqa_test500_equal_budget_bge_base",
    },
    "2wiki": {
        "label": "2WikiMultihopQA",
        "dataset_dir": "local-artifacts/datasets/ircot_2wikimultihopqa_test500",
        "main_retrievals": [
            ("Main", "Dense", "local-artifacts/results/ircot_2wiki_test500_top10_retrieval.jsonl"),
            ("Main", "BM25", "local-artifacts/results/ircot_2wiki_test500_bm25_top10_retrieval.jsonl"),
            ("Main", "Hybrid", "local-artifacts/results/ircot_2wiki_test500_hybrid_top10_retrieval.jsonl"),
            (
                "Main",
                "Hypothetical-only",
                "local-artifacts/results/ircot_2wiki_test500_hyde_hypothetical_only_top10_retrieval.jsonl",
            ),
            ("Main", "HyDE", "local-artifacts/results/ircot_2wiki_test500_hyde_top10_retrieval.jsonl"),
        ],
        "equal_budget_prefix": "ircot_2wiki_test500_equal_budget_bge_base",
    },
}


EQUAL_BUDGET_MODES = [
    ("Keyword/entity", "keyword_expansion"),
    ("Direct rewrite", "direct_rewrite"),
    ("Decomposition", "question_decomposition"),
    ("Document-like", "document_like_passage"),
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(value):
    return " ".join(str(value or "").split()).casefold()


def paragraph_key(doc):
    return (normalize_text(doc.get("title", "")), normalize_text(doc.get("text", "")))


def support_titles(row, fallback_titles=None):
    facts = row.get("supporting_facts", {})
    titles = []
    if isinstance(facts, dict):
        titles = facts.get("title", []) or []
    if not titles and fallback_titles:
        titles = list(fallback_titles)
    return {normalize_text(title) for title in titles if normalize_text(title)}


def build_support_index(docs):
    by_question = defaultdict(lambda: {"paragraph_keys": set(), "record_ids": set(), "titles": set()})
    for doc in docs:
        if not doc.get("is_supporting"):
            continue
        qid = doc.get("source_question_id")
        if not qid:
            continue
        by_question[qid]["paragraph_keys"].add(paragraph_key(doc))
        if doc.get("doc_id"):
            by_question[qid]["record_ids"].add(str(doc["doc_id"]))
        title = normalize_text(doc.get("title", ""))
        if title:
            by_question[qid]["titles"].add(title)
    return dict(by_question)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def retrieval_metrics(rows, support_index):
    rows = list(rows)
    counts = {
        "n": 0,
        "title_any": 0,
        "title_all": 0,
        "title_recall_sum": 0.0,
        "paragraph_any": 0,
        "paragraph_all": 0,
        "paragraph_recall_sum": 0.0,
        "record_any": 0,
        "record_all": 0,
        "record_recall_sum": 0.0,
        "title_all_not_paragraph_all": 0,
        "paragraph_all_not_title_all": 0,
        "title_paragraph_all_agree": 0,
        "gold_paragraph_sum": 0,
        "gold_record_sum": 0,
        "missing_support_annotation": 0,
    }

    for row in rows:
        qid = row.get("id")
        gold = support_index.get(qid, {"paragraph_keys": set(), "record_ids": set(), "titles": set()})
        gold_titles = support_titles(row, fallback_titles=gold["titles"])
        gold_paragraphs = set(gold["paragraph_keys"])
        gold_records = set(gold["record_ids"])

        retrieved = row.get("retrieved", [])
        retrieved_titles = {normalize_text(doc.get("title", "")) for doc in retrieved if normalize_text(doc.get("title", ""))}
        retrieved_paragraphs = {paragraph_key(doc) for doc in retrieved}
        retrieved_records = {str(doc.get("doc_id")) for doc in retrieved if doc.get("doc_id")}

        title_hits = gold_titles & retrieved_titles
        paragraph_hits = gold_paragraphs & retrieved_paragraphs
        record_hits = gold_records & retrieved_records

        title_all = bool(gold_titles) and gold_titles.issubset(retrieved_titles)
        paragraph_all = bool(gold_paragraphs) and gold_paragraphs.issubset(retrieved_paragraphs)
        record_all = bool(gold_records) and gold_records.issubset(retrieved_records)

        counts["n"] += 1
        counts["title_any"] += bool(title_hits)
        counts["title_all"] += title_all
        counts["title_recall_sum"] += safe_divide(len(title_hits), len(gold_titles))
        counts["paragraph_any"] += bool(paragraph_hits)
        counts["paragraph_all"] += paragraph_all
        counts["paragraph_recall_sum"] += safe_divide(len(paragraph_hits), len(gold_paragraphs))
        counts["record_any"] += bool(record_hits)
        counts["record_all"] += record_all
        counts["record_recall_sum"] += safe_divide(len(record_hits), len(gold_records))
        counts["title_all_not_paragraph_all"] += title_all and not paragraph_all
        counts["paragraph_all_not_title_all"] += paragraph_all and not title_all
        counts["title_paragraph_all_agree"] += title_all == paragraph_all
        counts["gold_paragraph_sum"] += len(gold_paragraphs)
        counts["gold_record_sum"] += len(gold_records)
        counts["missing_support_annotation"] += not gold_paragraphs

    n = counts["n"]
    title_all_count = counts["title_all"]
    return {
        "n": n,
        "title_any_hit_at10": safe_divide(counts["title_any"], n),
        "title_all_hit_at10": safe_divide(counts["title_all"], n),
        "title_recall_at10": safe_divide(counts["title_recall_sum"], n),
        "paragraph_any_hit_at10": safe_divide(counts["paragraph_any"], n),
        "paragraph_all_hit_at10": safe_divide(counts["paragraph_all"], n),
        "paragraph_recall_at10": safe_divide(counts["paragraph_recall_sum"], n),
        "record_any_hit_at10": safe_divide(counts["record_any"], n),
        "record_all_hit_at10": safe_divide(counts["record_all"], n),
        "record_recall_at10": safe_divide(counts["record_recall_sum"], n),
        "title_minus_paragraph_all": safe_divide(counts["title_all"] - counts["paragraph_all"], n),
        "title_all_not_paragraph_all": counts["title_all_not_paragraph_all"],
        "paragraph_all_not_title_all": counts["paragraph_all_not_title_all"],
        "title_paragraph_all_agreement": safe_divide(counts["title_paragraph_all_agree"], n),
        "paragraph_all_given_title_all": safe_divide(counts["paragraph_all"], title_all_count),
        "mean_gold_supporting_paragraphs": safe_divide(counts["gold_paragraph_sum"], n),
        "mean_gold_supporting_records": safe_divide(counts["gold_record_sum"], n),
        "missing_support_annotation": counts["missing_support_annotation"],
    }


def fmt(value, digits=4, signed=False):
    if isinstance(value, int):
        return str(value)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def metric_row(dataset, family, method, retrieval_path, support_index):
    rows = list(read_jsonl(retrieval_path))
    metrics = retrieval_metrics(rows, support_index)
    return {
        "dataset": dataset,
        "family": family,
        "method": method,
        "retrieval_file": str(retrieval_path),
        **metrics,
    }


def collect_rows(paper_root, include_equal_budget=True):
    paper_root = Path(paper_root)
    all_rows = []
    for dataset_key, cfg in DATASETS.items():
        dataset_dir = paper_root / cfg["dataset_dir"]
        support_index = build_support_index(read_jsonl(dataset_dir / "corpus.jsonl"))
        for family, method, rel_path in cfg["main_retrievals"]:
            all_rows.append(metric_row(cfg["label"], family, method, paper_root / rel_path, support_index))

        if include_equal_budget:
            retrieval_dir = paper_root / "local-artifacts/equal_budget_query_diagnostic/retrieval"
            for method, mode in EQUAL_BUDGET_MODES:
                path = retrieval_dir / f"{cfg['equal_budget_prefix']}_{mode}_top10_retrieval.jsonl"
                all_rows.append(metric_row(cfg["label"], "Equal-budget", method, path, support_index))
    return all_rows


def write_csv(path, rows):
    fieldnames = [
        "dataset",
        "family",
        "method",
        "n",
        "title_any_hit_at10",
        "title_all_hit_at10",
        "title_recall_at10",
        "paragraph_any_hit_at10",
        "paragraph_all_hit_at10",
        "paragraph_recall_at10",
        "record_any_hit_at10",
        "record_all_hit_at10",
        "record_recall_at10",
        "title_minus_paragraph_all",
        "title_all_not_paragraph_all",
        "paragraph_all_not_title_all",
        "title_paragraph_all_agreement",
        "paragraph_all_given_title_all",
        "mean_gold_supporting_paragraphs",
        "mean_gold_supporting_records",
        "missing_support_annotation",
        "retrieval_file",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "# Supporting Paragraph Metric Audit",
        "",
        "| Dataset | Family | Method | Title all@10 | Paragraph all@10 | Paragraph recall@10 | Title-paragraph all gap | Paragraph all given title all |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['family']} | {row['method']} | "
            f"{fmt(row['title_all_hit_at10'])} | "
            f"{fmt(row['paragraph_all_hit_at10'])} | "
            f"{fmt(row['paragraph_recall_at10'])} | "
            f"{fmt(row['title_minus_paragraph_all'], signed=True)} | "
            f"{fmt(row['paragraph_all_given_title_all'])} |"
        )
    lines.extend(
        [
            "",
            "Paragraph metrics use exact normalized `(title, paragraph text)` keys for gold `is_supporting=true` corpus records.",
            "In the released processed splits, each annotated supporting title is associated with one supporting paragraph record, so this audit primarily verifies exact processed-record recovery rather than introducing an independent sentence-level evidence notion.",
            "Record metrics in the CSV additionally require the retrieved source-question-specific `doc_id`; this is stricter than evidence equivalence when the same title--paragraph text appears in multiple local pools.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    selected = list(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Supporting-paragraph audit for reported title-level retrieval rows.}",
        r"\label{tab:supporting_paragraph_audit}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X l l r r r r}",
        r"\toprule",
        r"Dataset & Family & Method & Title all@10 & Paragraph all@10 & Paragraph recall@10 & Gap \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['dataset']} & {row['family']} & {row['method']} & "
            f"{fmt(row['title_all_hit_at10'])} & "
            f"{fmt(row['paragraph_all_hit_at10'])} & "
            f"{fmt(row['paragraph_recall_at10'])} & "
            f"{fmt(row['title_minus_paragraph_all'], signed=True)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item The audited rows are exactly the rows listed in the table. They include the HotpotQA MiniLM main-result, query-composition, heuristic-control retrieval rows, the 2WikiMultihopQA reported MiniLM retrieval rows with available artifacts, and the four equal-budget query-form rows for both datasets. Title all@10 matches the main all-support hit@10 metric: all annotated supporting titles must appear in the top-10 retrieved evidence. Paragraph all@10 is stricter: all gold corpus records marked \texttt{is\_supporting=true} for the question must be recovered as exact normalized title--paragraph text. The gap is title all@10 minus paragraph all@10.",
            r"\item In the released processed splits, each annotated supporting title is associated with one supporting paragraph record, so identical title-level and paragraph-level values are expected when the retrieved title corresponds to the same processed paragraph text. The processed artifacts do not expose gold supporting sentence text in the 500-example JSONL questions used here, so this audit cannot compute sentence-level supporting-fact coverage. It primarily verifies exact processed-record recovery rather than introducing an independent sentence-level evidence notion.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table*}",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--out_csv", default="local-artifacts/supporting_paragraph_audit/supporting_paragraph_metrics.csv")
    parser.add_argument("--out_md", default="local-artifacts/supporting_paragraph_audit/supporting_paragraph_metrics.md")
    parser.add_argument("--out_tex", default="paper/latex/table_supporting_paragraph_audit.tex")
    parser.add_argument("--no_equal_budget", action="store_true")
    args = parser.parse_args()

    paper_root = Path(args.paper_root)
    rows = collect_rows(paper_root, include_equal_budget=not args.no_equal_budget)
    write_csv(paper_root / args.out_csv, rows)
    write_markdown(paper_root / args.out_md, rows)
    write_latex(paper_root / args.out_tex, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
