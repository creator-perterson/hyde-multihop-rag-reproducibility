import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_answers import exact_match
from utils import read_jsonl, write_jsonl


VERIFICATION_TEMPLATE = """You are an evidence-grounded answer verification agent.

Given a question, retrieved evidence, and an initial short answer:
1. Check whether the initial answer is directly supported by the evidence.
2. Check whether the answer has the format requested by the question.
3. If the answer is unsupported, too broad, too narrow, or in the wrong format, correct it using only the evidence.
4. If the evidence is insufficient, set final_answer to "I don't know".

Return only valid JSON with this exact schema:
{{"verdict":"keep|correct|abstain","final_answer":"...","reason":"short reason"}}

Question:
{question}

Initial answer:
{prediction}

Evidence:
{evidence}

JSON:"""


CANONICAL_VERIFICATION_TEMPLATE = """You are an evidence-grounded answer verification and canonicalization agent.

Your goal is not only to check whether the initial answer is semantically plausible, but to rewrite it into the most exact short answer phrase expected by a multi-hop QA benchmark.

Given a question, retrieved evidence, and an initial answer:
1. Verify whether the answer is supported by evidence.
2. If supported but not canonical, rewrite it to the most specific exact phrase from the evidence.
3. Prefer full person names over shortened names when the evidence gives the full name.
4. For nationality questions, output the adjectival nationality when available, not only the country name. Example: "American" is preferred over "United States citizen".
5. For count/quantity questions, include the unit if the question asks for it. Example: "12 member universities" is preferred over "12".
6. For county questions, return the county name alone if the benchmark-style answer usually omits the word "County".
7. For date questions, include the year if the question asks for an event time and the evidence gives month plus year.
8. If the initial answer is wrong, unsupported, too broad, too narrow, or the evidence contains a better exact phrase, correct it.
9. If the evidence is insufficient, set final_answer to "I don't know".

Return only valid JSON with this exact schema:
{{"verdict":"keep|correct|abstain","final_answer":"...","reason":"short reason"}}

Question:
{question}

Initial answer:
{prediction}

Evidence:
{evidence}

JSON:"""


CONSERVATIVE_VERIFICATION_TEMPLATE = """You are a conservative evidence-grounded answer verification agent.

Your default behavior is to KEEP the initial answer. Rewrite it only when there is a clear benchmark-style formatting issue and the evidence directly supports the corrected phrase.

Allowed corrections:
1. Nationality questions: prefer adjectival nationality when the evidence supports it. Example: "American" instead of "United States citizen".
2. Count questions: if the initial answer is only a number but the question asks for a counted object, include the unit from the evidence. Example: "12 member universities" instead of "12".
3. Date questions: if the question asks for an event time and the evidence gives a more complete month/year phrase, include the complete phrase.
4. Yes/no questions: if the question asks a yes/no comparison and the evidence directly supports it, normalize the answer to "Yes" or "No".
5. Abstained answers: if the initial answer is "I don't know" but the evidence directly contains the exact requested short answer, provide that answer. Otherwise keep "I don't know".
6. Location, role, title, or type questions: only add a missing qualifier when the question asks for that qualifier and the exact phrase appears in evidence.

Do not make these changes:
1. Do not abstain if the initial answer is directly supported by any evidence.
2. Do not expand a familiar short person name to a full legal name unless the question explicitly asks for the full name.
3. Do not remove location qualifiers such as state/country names.
4. Do not remove parenthetical title explanations.
5. Do not rewrite a correct answer merely because another phrase is also present.
6. Do not add units to a numeric answer when the question only asks for a number.
7. Do not replace a precise answer with a longer phrase unless the longer phrase is explicitly requested by the question.

Return only valid JSON with this exact schema:
{{"verdict":"keep|correct|abstain","final_answer":"...","reason":"short reason"}}

Question:
{question}

Initial answer:
{prediction}

Evidence:
{evidence}

JSON:"""


def format_evidence(row, top_k, max_chars_per_doc):
    chunks = []
    for rank, doc in enumerate(row.get("retrieved", [])[:top_k], start=1):
        text = doc.get("text", "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        chunks.append(f"[{rank}] Title: {doc.get('title', '')}\n{text}")
    return "\n\n".join(chunks)


def _normalized_prediction(row):
    return str(row.get("prediction", "")).strip().lower().rstrip(".")


def _token_count(text):
    return len(str(text).split())


def risk_reasons(row):
    question = row.get("question", "").lower()
    prediction = _normalized_prediction(row)
    first_word = question.split(" ", 1)[0] if question else ""
    reasons = []

    if prediction.startswith("i don") or prediction in {"unknown", "not enough information"}:
        reasons.append(("abstained_answer", 100))

    if "nationality" in question:
        reasons.append(("nationality_format", 95))

    date_cues = [
        "what month",
        "what year",
        "what date",
        "when did",
        "when was",
        "when were",
        "when is",
    ]
    if any(cue in question for cue in date_cues) and _token_count(prediction) <= 2:
        reasons.append(("date_granularity", 90))

    if question.startswith("how many") and prediction.isdigit():
        reasons.append(("bare_count", 85))

    if first_word in {"are", "is", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had"} and prediction not in {"yes", "no"}:
        reasons.append(("yes_no_normalization", 80))

    location_cues = [
        "what city",
        "which city",
        "what state",
        "which state",
        "what country",
        "which country",
        "what county",
        "which county",
        "what neighborhood",
        "which neighborhood",
    ]
    if any(cue in question for cue in location_cues) and 1 <= _token_count(prediction) <= 4:
        reasons.append(("location_granularity", 55))

    role_cues = [
        "what government position",
        "what position",
        "which position",
        "what office",
        "which office",
        "what role",
        "what profession",
        "what occupation",
        "what type",
        "what kind",
    ]
    if any(cue in question for cue in role_cues) and 1 <= _token_count(prediction) <= 5:
        reasons.append(("role_or_type_granularity", 50))

    comparison_cues = ["who is older", "who was older", "who is younger", "who was younger"]
    if any(cue in question for cue in comparison_cues) and 1 <= _token_count(prediction) <= 4:
        reasons.append(("comparison_entity_check", 45))

    if _token_count(prediction) >= 8:
        reasons.append(("overlong_answer", 35))

    if first_word in {"who", "what", "which"} and 1 <= _token_count(prediction) <= 2:
        reasons.append(("short_entity_granularity", 20))

    return reasons


def risk_score(row):
    reasons = risk_reasons(row)
    if not reasons:
        return 0
    return max(score for _, score in reasons)


def is_legacy_risk_candidate(row):
    question = row.get("question", "").lower()
    prediction = _normalized_prediction(row)
    if "nationality" in question:
        return True
    if question.startswith("how many") and prediction.isdigit():
        return True
    if ("what month" in question or "what year" in question) and _token_count(prediction) == 1:
        return True
    return False


def is_risk_candidate(row):
    return risk_score(row) > 0


def select_rows(rows, limit, strategy, risk_target=120):
    if strategy == "errors":
        selected = [row for row in rows if not exact_match(row["prediction"], row["gold_answer"])]
    elif strategy == "risk":
        legacy = [
            (index, row)
            for index, row in enumerate(rows)
            if is_legacy_risk_candidate(row)
        ]
        legacy_ids = {id(row) for _, row in legacy}
        scored_extra = [
            (risk_score(row), index, row)
            for index, row in enumerate(rows)
            if is_risk_candidate(row) and id(row) not in legacy_ids
        ]
        scored_extra.sort(key=lambda item: (-item[0], item[1]))
        target = limit if limit else risk_target
        selected = [row for _, row in legacy] + [row for _, _, row in scored_extra]
        selected = selected[:target]
    elif strategy == "balanced":
        wrong = [row for row in rows if not exact_match(row["prediction"], row["gold_answer"])]
        correct = [row for row in rows if exact_match(row["prediction"], row["gold_answer"])]
        half = limit // 2 if limit else len(wrong)
        selected = wrong[:half] + correct[: max(0, (limit or 0) - min(half, len(wrong)))]
    else:
        selected = rows
    return selected[:limit] if limit else selected


def build_prompt_row(row, top_k, max_chars_per_doc, style):
    if style == "canonical":
        template = CANONICAL_VERIFICATION_TEMPLATE
    elif style == "conservative":
        template = CONSERVATIVE_VERIFICATION_TEMPLATE
    else:
        template = VERIFICATION_TEMPLATE
    prompt = template.format(
        question=row["question"],
        prediction=row["prediction"],
        evidence=format_evidence(row, top_k, max_chars_per_doc),
    )
    return {
        "id": row["id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "supporting_facts": row.get("supporting_facts", {}),
        "retrieved": row.get("retrieved", [])[:top_k],
        "initial_prediction": row["prediction"],
        "risk_score": risk_score(row),
        "risk_reasons": [reason for reason, _ in risk_reasons(row)],
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_strategy", choices=["all", "errors", "balanced", "risk"], default="balanced")
    parser.add_argument("--risk_target", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_chars_per_doc", type=int, default=900)
    parser.add_argument("--style", choices=["standard", "canonical", "conservative"], default="standard")
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    selected = select_rows(rows, args.limit, args.sample_strategy, args.risk_target)
    prompt_rows = [build_prompt_row(row, args.top_k, args.max_chars_per_doc, args.style) for row in selected]
    write_jsonl(args.out, prompt_rows)
    print(f"Saved {len(prompt_rows)} verification prompts to {args.out}")


if __name__ == "__main__":
    main()
