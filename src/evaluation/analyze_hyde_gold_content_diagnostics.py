import argparse
import csv
import math
import random
import re
import string
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import f1_score
from utils import read_jsonl, write_jsonl


DATASETS = {
    "hotpotqa": {
        "display": "HotpotQA",
        "questions": "datasets/ircot_hotpotqa_test500/questions.jsonl",
        "index_dir": "datasets/ircot_hotpotqa_test500/faiss_index",
        "hyde_generation": "results/ircot_hotpotqa_test500_hyde_generation_qwenmax_500.jsonl",
        "dense_retrieval": "results/ircot_hotpotqa_test500_top10_retrieval.jsonl",
        "hyde_retrieval": "results/ircot_hotpotqa_test500_hyde_top10_retrieval.jsonl",
        "dense_answers": "results/ircot_hotpotqa_test500_top10_extractive_answers_qwen_500.jsonl",
        "hyde_answers": "results/ircot_hotpotqa_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
        "artifact_prefix": "ircot_hotpotqa_test500",
    },
    "2wiki": {
        "display": "2WikiMultihopQA",
        "questions": "datasets/ircot_2wikimultihopqa_test500/questions.jsonl",
        "index_dir": "datasets/ircot_2wikimultihopqa_test500/faiss_index",
        "hyde_generation": "results/ircot_2wiki_test500_hyde_generation_qwenmax_500.jsonl",
        "dense_retrieval": "results/ircot_2wiki_test500_top10_retrieval.jsonl",
        "hyde_retrieval": "results/ircot_2wiki_test500_hyde_top10_retrieval.jsonl",
        "dense_answers": "results/ircot_2wiki_test500_dense_top10_extractive_answers_qwenmax_500.jsonl",
        "hyde_answers": "results/ircot_2wiki_test500_hyde_top10_extractive_answers_qwenmax_500.jsonl",
        "artifact_prefix": "ircot_2wiki_test500",
    },
}

COMMON_ALIASES = {
    "american": ["united states", "u s", "usa", "us citizen", "united states citizen"],
    "united states": ["american", "u s", "usa", "us"],
    "u s": ["united states", "american", "usa"],
    "usa": ["united states", "american"],
    "british": ["united kingdom", "uk", "english"],
    "united kingdom": ["british", "uk"],
    "english": ["england", "british"],
    "french": ["france"],
    "german": ["germany"],
    "italian": ["italy"],
    "spanish": ["spain"],
    "canadian": ["canada"],
    "australian": ["australia"],
    "irish": ["ireland"],
    "scottish": ["scotland"],
    "welsh": ["wales"],
    "russian": ["russia"],
    "chinese": ["china"],
    "japanese": ["japan"],
    "indian": ["india"],
    "mexican": ["mexico"],
    "brazilian": ["brazil"],
    "argentine": ["argentina"],
    "dutch": ["netherlands", "holland"],
    "south korean": ["korea", "republic of korea"],
    "korean": ["south korea", "north korea"],
}

GROUP_ORDER = [
    "Exact-answer present",
    "Alias/paraphrase proxy present",
    "Supporting-entity present",
    "No identifiable gold content",
]


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    return " ".join(text.split())


def normalize_exact_overlap(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def content_tokens(text):
    return [tok for tok in normalize_text(text).split() if tok not in {"of", "in", "on", "and", "or", "to", "for"}]


def contains_phrase(text, phrase):
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)
    if not phrase_norm or not text_norm:
        return False
    pattern = r"(?:^|\s)" + re.escape(phrase_norm) + r"(?:\s|$)"
    return re.search(pattern, text_norm) is not None


def contains_normalized_substring(text, phrase):
    phrase_norm = normalize_exact_overlap(phrase)
    text_norm = normalize_exact_overlap(text)
    return bool(phrase_norm) and bool(text_norm) and phrase_norm in text_norm


def stripped_parenthetical(text):
    return re.sub(r"\s*\([^)]*\)", "", str(text)).strip()


def surface_variants(text):
    variants = {str(text).strip(), stripped_parenthetical(text)}
    raw = str(text)
    for sep in ["/", ";", ":", " - ", " -- "]:
        for part in raw.split(sep):
            part = part.strip()
            if part:
                variants.add(part)
                variants.add(stripped_parenthetical(part))
    return {v for v in variants if normalize_text(v)}


def answer_alias_variants(answer):
    variants = set()
    answer_norm = normalize_text(answer)
    for variant in surface_variants(answer):
        if normalize_text(variant) != answer_norm:
            variants.add(variant)
    variants.update(COMMON_ALIASES.get(answer_norm, []))
    return {v for v in variants if normalize_text(v) and normalize_text(v) != answer_norm}


def has_paraphrase_proxy(answer, document):
    aliases = answer_alias_variants(answer)
    if any(contains_phrase(document, alias) for alias in aliases):
        return True
    gold_tokens = content_tokens(answer)
    if len(gold_tokens) >= 3:
        doc_tokens = set(content_tokens(document))
        overlap = sum(1 for tok in gold_tokens if tok in doc_tokens)
        return overlap >= max(2, math.ceil(0.8 * len(gold_tokens)))
    return False


def support_title_variants(title):
    variants = surface_variants(title)
    normalized = normalize_text(title)
    if "," in str(title):
        variants.add(str(title).split(",", 1)[0].strip())
    return {v for v in variants if normalize_text(v) and len(normalize_text(v)) >= 3 and normalize_text(v) != normalized[:1]}


def support_entity_hits(titles, document):
    hits = []
    for title in titles:
        if any(contains_phrase(document, variant) for variant in support_title_variants(title)):
            hits.append(title)
    return sorted(set(hits))


def classify_gold_content(row):
    answer = row.get("gold_answer", row.get("answer", ""))
    document = row.get("prediction", row.get("hyde_document", ""))
    question = row.get("question", "")
    titles = row.get("supporting_facts", {}).get("title", [])

    exact_answer = contains_normalized_substring(document, answer)
    alias_proxy = (not exact_answer) and has_paraphrase_proxy(answer, document)
    title_hits = support_entity_hits(titles, document)
    title_hits_not_in_question = [
        title for title in title_hits
        if not any(contains_phrase(question, variant) for variant in support_title_variants(title))
    ]

    if exact_answer:
        group = "Exact-answer present"
    elif alias_proxy:
        group = "Alias/paraphrase proxy present"
    elif title_hits:
        group = "Supporting-entity present"
    else:
        group = "No identifiable gold content"

    return {
        "gold_content_group": group,
        "exact_answer_present": int(exact_answer),
        "alias_paraphrase_proxy_present": int(alias_proxy),
        "supporting_title_present": int(bool(title_hits)),
        "supporting_title_not_in_question_present": int(bool(title_hits_not_in_question)),
        "supporting_title_hits": "; ".join(title_hits),
        "supporting_title_hits_not_in_question": "; ".join(title_hits_not_in_question),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    index = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def bootstrap_ci(deltas, iterations=2000, seed=13):
    deltas = list(deltas)
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        estimates.append(mean(sample))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def load_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def support_metrics(row):
    gold_titles = set(row.get("supporting_facts", {}).get("title", []))
    retrieved_titles = {doc.get("title", "") for doc in row.get("retrieved", [])}
    hits = gold_titles & retrieved_titles
    return {
        "any_hit": float(bool(hits)),
        "all_hit": float(gold_titles.issubset(retrieved_titles) if gold_titles else False),
        "recall": len(hits) / len(gold_titles) if gold_titles else 0.0,
    }


def answer_f1(row):
    return f1_score(row.get("prediction", row.get("final_answer", "")), row.get("gold_answer", row.get("answer", "")))


def group_summary(dataset, group, ids, dense_retrieval, hyde_retrieval, dense_answers, hyde_answers, seed):
    ids = [qid for qid in ids if qid in dense_retrieval and qid in hyde_retrieval and qid in dense_answers and qid in hyde_answers]
    row = {"dataset": dataset, "group": group, "n": len(ids)}
    for metric in ["all_hit", "recall", "any_hit"]:
        dense_values = [support_metrics(dense_retrieval[qid])[metric] for qid in ids]
        hyde_values = [support_metrics(hyde_retrieval[qid])[metric] for qid in ids]
        deltas = [h - d for d, h in zip(dense_values, hyde_values)]
        low, high = bootstrap_ci(deltas, seed=seed + len(metric))
        row[f"dense_{metric}"] = mean(dense_values)
        row[f"hyde_{metric}"] = mean(hyde_values)
        row[f"delta_{metric}"] = mean(deltas)
        row[f"delta_{metric}_ci_low"] = low
        row[f"delta_{metric}_ci_high"] = high
    dense_f1 = [answer_f1(dense_answers[qid]) for qid in ids]
    hyde_f1 = [answer_f1(hyde_answers[qid]) for qid in ids]
    f1_deltas = [h - d for d, h in zip(dense_f1, hyde_f1)]
    low, high = bootstrap_ci(f1_deltas, seed=seed + 101)
    row["dense_f1"] = mean(dense_f1)
    row["hyde_f1"] = mean(hyde_f1)
    row["delta_f1"] = mean(f1_deltas)
    row["delta_f1_ci_low"] = low
    row["delta_f1_ci_high"] = high
    return row


def flexible_phrase_pattern(phrase):
    tokens = content_tokens(phrase)
    if not tokens:
        return None
    return re.compile(r"\b" + r"[\W_]+".join(re.escape(tok) for tok in tokens) + r"\b", flags=re.IGNORECASE)


def scrub_phrases(text, phrases, replacement="[REDACTED]"):
    scrubbed = str(text)
    for phrase in sorted(phrases, key=lambda p: len(str(p)), reverse=True):
        pattern = flexible_phrase_pattern(phrase)
        if pattern is not None:
            scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


def scrubbed_hyde_document(row, mode):
    document = row.get("prediction", "")
    answer = row.get("gold_answer", row.get("answer", ""))
    titles = row.get("supporting_facts", {}).get("title", [])
    phrases = set(surface_variants(answer))
    if mode == "gold_content_scrubbed":
        phrases.update(answer_alias_variants(answer))
        for title in titles:
            phrases.update(support_title_variants(title))
    return scrub_phrases(document, phrases)


def build_query(question, hyde_document, max_hyde_chars=900):
    hyde_document = hyde_document.strip()
    if len(hyde_document) > max_hyde_chars:
        hyde_document = hyde_document[:max_hyde_chars].rstrip() + "..."
    return f"{question.strip()}\n\nHypothetical supporting passage:\n{hyde_document}".strip()


def format_retrieved(score_row, index_row, docs):
    retrieved = []
    for score, doc_idx in zip(score_row, index_row):
        doc = docs[int(doc_idx)]
        retrieved.append({
            "score": float(score),
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": doc["text"],
            "source_question_id": doc["source_question_id"],
        })
    return retrieved


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts, tokenizer, model, batch_size):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def search_topk(query_embeddings, doc_embeddings, top_k):
    all_scores = []
    all_indices = []
    doc_t = doc_embeddings.t()
    for start in range(0, query_embeddings.shape[0], 64):
        scores = torch.mm(query_embeddings[start:start + 64], doc_t)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        all_scores.extend(top_scores.tolist())
        all_indices.extend(top_indices.tolist())
    return all_scores, all_indices


def run_scrubbed_retrieval(code_root, spec, mode, model_name, top_k, batch_size, out_path):
    questions = load_by_id(code_root / spec["questions"])
    hyde_rows = list(read_jsonl(code_root / spec["hyde_generation"]))
    docs = list(read_jsonl(code_root / spec["index_dir"] / Path("docstore.jsonl")))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)

    query_texts = [
        build_query(questions[row["id"]]["question"], scrubbed_hyde_document(row, mode))
        for row in hyde_rows
    ]
    doc_texts = [f"{doc['title']}. {doc['text']}" for doc in docs]
    doc_embeddings = encode_texts(doc_texts, tokenizer, model, batch_size)
    query_embeddings = encode_texts(query_texts, tokenizer, model, batch_size)
    scores, indices = search_topk(query_embeddings, doc_embeddings, top_k)

    out_rows = []
    for hyde_row, score_row, index_row in zip(hyde_rows, scores, indices):
        question = questions[hyde_row["id"]]
        out_rows.append({
            "id": question["id"],
            "question": question["question"],
            "answer": question["answer"],
            "supporting_facts": question.get("supporting_facts", {}),
            "retrieved": format_retrieved(score_row, index_row, docs),
            "hyde_document": hyde_row.get("prediction", ""),
            "scrubbed_hyde_document": scrubbed_hyde_document(hyde_row, mode),
            "retrieval_strategy": {
                "name": f"hyde_{mode}",
                "top_k": top_k,
                "model": model_name,
                "diagnostic_only": True,
            },
        })
    write_jsonl(out_path, out_rows)
    return out_rows


def retrieval_summary(dataset, method, rows, baseline_rows=None, original_hyde_rows=None, seed=13):
    ids = [row["id"] for row in rows]
    by_id = {row["id"]: row for row in rows}
    row = {
        "dataset": dataset,
        "method": method,
        "n": len(ids),
        "all_hit": mean([support_metrics(by_id[qid])["all_hit"] for qid in ids]),
        "recall": mean([support_metrics(by_id[qid])["recall"] for qid in ids]),
        "any_hit": mean([support_metrics(by_id[qid])["any_hit"] for qid in ids]),
    }
    if baseline_rows is not None:
        baseline = {r["id"]: r for r in baseline_rows}
        paired = [qid for qid in ids if qid in baseline]
        deltas = [support_metrics(by_id[qid])["all_hit"] - support_metrics(baseline[qid])["all_hit"] for qid in paired]
        low, high = bootstrap_ci(deltas, seed=seed + 17)
        row["delta_all_vs_dense"] = mean(deltas)
        row["delta_all_vs_dense_ci_low"] = low
        row["delta_all_vs_dense_ci_high"] = high
    if original_hyde_rows is not None:
        original = {r["id"]: r for r in original_hyde_rows}
        paired = [qid for qid in ids if qid in original]
        deltas = [support_metrics(by_id[qid])["all_hit"] - support_metrics(original[qid])["all_hit"] for qid in paired]
        low, high = bootstrap_ci(deltas, seed=seed + 29)
        row["delta_all_vs_original_hyde"] = mean(deltas)
        row["delta_all_vs_original_hyde_ci_low"] = low
        row["delta_all_vs_original_hyde_ci_high"] = high
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{float(value):.4f}"


def ci_text(row, prefix):
    return f"{fmt(row[prefix])} [{fmt(row[prefix + '_ci_low'])}, {fmt(row[prefix + '_ci_high'])}]"


def write_markdown(path, group_rows, scrub_rows):
    lines = [
        "# HyDE Gold-Content Diagnostics",
        "",
        "The grouping diagnostic is an automatic lexical/entity diagnostic. It is designed to make answer-leakage risk more visible, not to prove that no paraphrased or memorized information remains.",
        "",
        "## Mutually Exclusive Gold-Content Groups",
        "",
        "| Dataset | Group | N | Dense all | HyDE all | Delta all [95% CI] | Dense recall | HyDE recall | Dense F1 | HyDE F1 | Delta F1 [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['dataset']} | {row['group']} | {row['n']} | {fmt(row['dense_all_hit'])} | {fmt(row['hyde_all_hit'])} | "
            f"{ci_text(row, 'delta_all_hit')} | {fmt(row['dense_recall'])} | {fmt(row['hyde_recall'])} | "
            f"{fmt(row['dense_f1'])} | {fmt(row['hyde_f1'])} | {ci_text(row, 'delta_f1')} |"
        )
    lines.extend([
        "",
        "## Gold-Aware Scrubbed Retrieval-Only Diagnostic",
        "",
        "| Dataset | Method | All hit@10 | Recall@10 | Delta all vs Dense [95% CI] | Delta all vs original HyDE [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in scrub_rows:
        dense_delta = ci_text(row, "delta_all_vs_dense") if "delta_all_vs_dense" in row else ""
        hyde_delta = ci_text(row, "delta_all_vs_original_hyde") if "delta_all_vs_original_hyde" in row else ""
        lines.append(
            f"| {row['dataset']} | {row['method']} | {fmt(row['all_hit'])} | {fmt(row['recall'])} | {dense_delta} | {hyde_delta} |"
        )
    lines.extend([
        "",
        "The scrubbed variants use gold-aware deletion only as offline diagnostics. They are not deployable inference modules because gold answers and supporting titles are unknown at test time.",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def latex_dataset_name(name):
    return "2Wiki" if str(name) == "2WikiMultihopQA" else str(name)


def write_latex(path, group_rows, scrub_rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Automatic gold-content stratification of HyDE hypothetical passages.}",
        r"\label{tab:hyde_gold_content_groups}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.10\textwidth} >{\raggedright\arraybackslash}X r r r r r r r r r}",
        r"\toprule",
        r"Dataset & Stratum & $N$ & Dense all & HyDE all & $\Delta$ all [95\% CI] & Dense recall & HyDE recall & Dense F1 & HyDE F1 & $\Delta$F1 [95\% CI] \\",
        r"\midrule",
    ]
    for row in group_rows:
        lines.append(
            f"{latex_escape(latex_dataset_name(row['dataset']))} & {latex_escape(row['group'])} & {row['n']} & "
            f"{fmt(row['dense_all_hit'])} & {fmt(row['hyde_all_hit'])} & {ci_text(row, 'delta_all_hit')} & "
            f"{fmt(row['dense_recall'])} & {fmt(row['hyde_recall'])} & "
            f"{fmt(row['dense_f1'])} & {fmt(row['hyde_f1'])} & {ci_text(row, 'delta_f1')} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item The strata are mutually exclusive and assigned in the order shown: exact normalized gold answer, automatic alias/paraphrase proxy, supporting-title/entity mention, and no identifiable gold content. The proxy is lexical and conservative; it does not replace human leakage annotation.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Gold-aware scrubbed HyDE retrieval-only diagnostic.}",
        r"\label{tab:hyde_gold_content_scrubbed}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{0.14\textwidth} >{\raggedright\arraybackslash}X r r r r}",
        r"\toprule",
        r"Dataset & Retrieval query & All hit@10 & Recall@10 & $\Delta$ all vs Dense [95\% CI] & $\Delta$ all vs original HyDE [95\% CI] \\",
        r"\midrule",
    ])
    for row in scrub_rows:
        dense_delta = ci_text(row, "delta_all_vs_dense") if "delta_all_vs_dense" in row else "--"
        hyde_delta = ci_text(row, "delta_all_vs_original_hyde") if "delta_all_vs_original_hyde" in row else "--"
        lines.append(
            f"{latex_escape(latex_dataset_name(row['dataset']))} & {latex_escape(row['method'])} & {fmt(row['all_hit'])} & {fmt(row['recall'])} & "
            f"{dense_delta} & {hyde_delta} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\begin{tablenotes}",
        r"\footnotesize",
        r"\item The scrubbed rows are offline diagnostics that delete gold-aware strings from the generated hypothetical passage before retrieval. They are not deployable inference methods because gold answers and supporting titles are unavailable at test time. Reader answers are not regenerated for scrubbed rows.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_dataset(code_root, spec_key, args):
    spec = DATASETS[spec_key]
    dataset = spec["display"]
    hyde_generation = list(read_jsonl(code_root / spec["hyde_generation"]))
    dense_retrieval = load_by_id(code_root / spec["dense_retrieval"])
    hyde_retrieval = load_by_id(code_root / spec["hyde_retrieval"])
    dense_answers = load_by_id(code_root / spec["dense_answers"])
    hyde_answers = load_by_id(code_root / spec["hyde_answers"])

    audit_rows = []
    ids_by_group = {group: [] for group in GROUP_ORDER}
    for row in hyde_generation:
        classification = classify_gold_content(row)
        audit_row = {
            "dataset": dataset,
            "id": row["id"],
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", row.get("answer", "")),
            "supporting_titles": "; ".join(row.get("supporting_facts", {}).get("title", [])),
            **classification,
        }
        audit_rows.append(audit_row)
        ids_by_group[classification["gold_content_group"]].append(row["id"])

    group_rows = [
        group_summary(
            dataset,
            group,
            ids_by_group[group],
            dense_retrieval,
            hyde_retrieval,
            dense_answers,
            hyde_answers,
            seed=args.seed + offset * 100,
        )
        for offset, group in enumerate(GROUP_ORDER)
    ]

    dense_rows = list(dense_retrieval.values())
    hyde_rows = list(hyde_retrieval.values())
    scrub_rows = [
        retrieval_summary(dataset, "Dense question only", dense_rows, baseline_rows=dense_rows, original_hyde_rows=hyde_rows, seed=args.seed),
        retrieval_summary(dataset, "Original HyDE q+h", hyde_rows, baseline_rows=dense_rows, original_hyde_rows=hyde_rows, seed=args.seed),
    ]

    for mode, label in [
        ("answer_scrubbed", "Answer-scrubbed HyDE q+h"),
        ("gold_content_scrubbed", "Answer+support-title-scrubbed HyDE q+h"),
    ]:
        out_path = code_root / args.scrub_artifact_dir / f"{spec['artifact_prefix']}_hyde_{mode}_top10_retrieval.jsonl"
        if out_path.exists() and not args.force_scrub:
            rows = list(read_jsonl(out_path))
        else:
            rows = run_scrubbed_retrieval(
                code_root=code_root,
                spec=spec,
                mode=mode,
                model_name=args.model,
                top_k=args.top_k,
                batch_size=args.batch_size,
                out_path=out_path,
            )
        scrub_rows.append(
            retrieval_summary(dataset, label, rows, baseline_rows=dense_rows, original_hyde_rows=hyde_rows, seed=args.seed)
        )

    return audit_rows, group_rows, scrub_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki"], choices=sorted(DATASETS))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force_scrub", action="store_true")
    parser.add_argument("--out_audit_jsonl", default="local-artifacts/hyde_gold_content_audit.jsonl")
    parser.add_argument("--out_group_csv", default="local-artifacts/hyde_gold_content_group_diagnostics.csv")
    parser.add_argument("--out_scrub_csv", default="local-artifacts/hyde_gold_content_scrubbed_retrieval.csv")
    parser.add_argument("--out_md", default="local-artifacts/hyde_gold_content_diagnostics.md")
    parser.add_argument("--out_tex", default="paper/latex/table_hyde_gold_content_diagnostics.tex")
    parser.add_argument("--scrub_artifact_dir", default="local-artifacts/gold_content_scrubbed_retrieval_artifacts")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    all_audit_rows = []
    all_group_rows = []
    all_scrub_rows = []
    for dataset in args.datasets:
        audit_rows, group_rows, scrub_rows = analyze_dataset(code_root, dataset, args)
        all_audit_rows.extend(audit_rows)
        all_group_rows.extend(group_rows)
        all_scrub_rows.extend(scrub_rows)

    write_jsonl(code_root / args.out_audit_jsonl, all_audit_rows)
    write_csv(code_root / args.out_group_csv, all_group_rows)
    write_csv(code_root / args.out_scrub_csv, all_scrub_rows)
    write_markdown(code_root / args.out_md, all_group_rows, all_scrub_rows)
    write_latex(code_root / args.out_tex, all_group_rows, all_scrub_rows)

    for row in all_group_rows:
        print(
            f"{row['dataset']} | {row['group']} | n={row['n']} | "
            f"Dense all={fmt(row['dense_all_hit'])} HyDE all={fmt(row['hyde_all_hit'])} "
            f"Delta all={ci_text(row, 'delta_all_hit')} Delta F1={ci_text(row, 'delta_f1')}"
        )
    for row in all_scrub_rows:
        print(
            f"{row['dataset']} | {row['method']} | all={fmt(row['all_hit'])} recall={fmt(row['recall'])}"
        )


if __name__ == "__main__":
    main()
