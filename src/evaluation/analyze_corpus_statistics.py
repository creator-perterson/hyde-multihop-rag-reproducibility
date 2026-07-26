import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = [
    ("HotpotQA", Path("datasets/ircot_hotpotqa_test500")),
    ("2Wiki", Path("datasets/ircot_2wikimultihopqa_test500")),
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fmt(value, digits=1):
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def dataset_stats(root):
    questions = list(read_jsonl(root / "questions.jsonl"))
    docs = list(read_jsonl(root / "corpus.jsonl"))

    docs_by_question = defaultdict(list)
    for doc in docs:
        docs_by_question[doc.get("source_question_id")].append(doc)

    candidate_counts = [len(docs_by_question[q["id"]]) for q in questions]
    support_title_counts = [
        len(set(q.get("supporting_facts", {}).get("title", []))) for q in questions
    ]
    support_doc_counts = [
        sum(1 for doc in docs_by_question[q["id"]] if doc.get("is_supporting"))
        for q in questions
    ]
    distractor_counts = [
        sum(1 for doc in docs_by_question[q["id"]] if not doc.get("is_supporting"))
        for q in questions
    ]

    return {
        "questions": len(questions),
        "total_candidate_paragraphs": len(docs),
        "unique_titles": len({doc.get("title", "") for doc in docs}),
        "mean_candidates_per_question": statistics.mean(candidate_counts),
        "median_candidates_per_question": statistics.median(candidate_counts),
        "min_candidates_per_question": min(candidate_counts),
        "max_candidates_per_question": max(candidate_counts),
        "mean_gold_supporting_titles": statistics.mean(support_title_counts),
        "median_gold_supporting_titles": statistics.median(support_title_counts),
        "mean_supporting_paragraphs": statistics.mean(support_doc_counts),
        "mean_distractor_paragraphs": statistics.mean(distractor_counts),
        "median_distractor_paragraphs": statistics.median(distractor_counts),
        "questions_with_candidate_pool": sum(1 for q in questions if q["id"] in docs_by_question),
    }


def write_csv(stats_by_dataset, out_csv):
    rows = []
    for dataset, stats in stats_by_dataset.items():
        row = {"dataset": dataset}
        row.update(stats)
        rows.append(row)

    fields = ["dataset"] + list(next(iter(stats_by_dataset.values())).keys())
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(stats_by_dataset, out_tex):
    hotpot = stats_by_dataset["HotpotQA"]
    twowiki = stats_by_dataset["2Wiki"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Local candidate-corpus statistics for the released processed splits.}",
        r"\label{tab:corpus_statistics}",
        r"\small",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X r r}",
        r"\toprule",
        r"Statistic & HotpotQA & 2WikiMultihopQA \\",
        r"\midrule",
        f"Questions & {fmt(hotpot['questions'])} & {fmt(twowiki['questions'])} " + r"\\",
        f"Total candidate paragraphs & {fmt(hotpot['total_candidate_paragraphs'])} & {fmt(twowiki['total_candidate_paragraphs'])} " + r"\\",
        f"Unique titles & {fmt(hotpot['unique_titles'])} & {fmt(twowiki['unique_titles'])} " + r"\\",
        (
            "Mean / median source-pool records contributed per question & "
            f"{fmt(hotpot['mean_candidates_per_question'])} / {fmt(hotpot['median_candidates_per_question'], 0)} & "
            f"{fmt(twowiki['mean_candidates_per_question'])} / {fmt(twowiki['median_candidates_per_question'], 0)} "
            + r"\\"
        ),
        f"Source-pool records contributed per question, min--max & {fmt(hotpot['min_candidates_per_question'])}--{fmt(hotpot['max_candidates_per_question'])} & {fmt(twowiki['min_candidates_per_question'])}--{fmt(twowiki['max_candidates_per_question'])} " + r"\\",
        f"Mean gold supporting titles & {float(hotpot['mean_gold_supporting_titles']):.2f} & {float(twowiki['mean_gold_supporting_titles']):.2f} " + r"\\",
        f"Mean supporting paragraphs & {float(hotpot['mean_supporting_paragraphs']):.2f} & {float(twowiki['mean_supporting_paragraphs']):.2f} " + r"\\",
        f"Mean / median distractor paragraphs & {fmt(hotpot['mean_distractor_paragraphs'])} / {fmt(hotpot['median_distractor_paragraphs'], 0)} & {fmt(twowiki['mean_distractor_paragraphs'])} / {fmt(twowiki['median_distractor_paragraphs'], 0)} " + r"\\",
        r"Corpus organization & Joint index over local pools & Joint index over local pools \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item Source-pool records are the records contributed by each released question-level candidate pool before all records are unioned into a dataset-level joint index. Retrieval is performed over the full joint corpus, not over a separate per-question pool. Exact duplicate title--paragraph records are retained before indexing; if the same paragraph appears in multiple question-level pools, each occurrence remains a separate document with a source-question-specific document identifier. Dense, BM25, and hybrid retrieval encode only document titles and paragraph texts. Question identifiers, supporting-evidence flags, gold answers, and evaluation metadata are retained in JSONL artifacts for analysis but are excluded from retrieval representations.",
        r"\item In both released processed splits, each annotated supporting title is associated with exactly one gold supporting paragraph record for every question, which explains why the mean supporting-title and supporting-paragraph counts are identical.",
        r"\item Supporting-title metrics collapse duplicate retrieved titles before computing any-hit@10, all-hit@10, and supporting-title recall@10.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
        "",
    ]
    Path(out_tex).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_root", default=".")
    parser.add_argument("--out_csv", default="local-artifacts/corpus_statistics_test500.csv")
    parser.add_argument("--out_tex", default="paper/latex/table_corpus_statistics.tex")
    args = parser.parse_args()

    code_root = Path(args.code_root)
    stats_by_dataset = {
        name: dataset_stats(code_root / dataset_root)
        for name, dataset_root in DATASETS
    }

    write_csv(stats_by_dataset, code_root / args.out_csv)
    write_latex(stats_by_dataset, code_root / args.out_tex)

    for name, stats in stats_by_dataset.items():
        print(name)
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
