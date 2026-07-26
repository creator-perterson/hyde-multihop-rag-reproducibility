import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import read_jsonl, write_jsonl


DEFAULT_SYSTEM_PROMPT = "You answer questions with short factual answers based only on the provided evidence."


def load_project_env():
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    load_dotenv(env_path)
    return env_path


def get_client(timeout):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set LLM_API_KEY or OPENAI_API_KEY in your terminal."
        )
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return OpenAI(api_key=api_key, timeout=timeout)


def build_messages(row, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": row["prompt"]},
    ]


def create_completion_with_retries(client, args, row):
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            return client.chat.completions.create(
                model=args.model,
                messages=build_messages(row, system_prompt=args.system_prompt),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt == args.retries:
                break
            print(
                f"Request failed for id={row['id']} on attempt {attempt}/{args.retries}: {exc}. Retrying...",
                file=sys.stderr,
            )
            time.sleep(args.retry_sleep)
    raise last_error


def build_output_row(row, prediction, model):
    output = {
        "id": row["id"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "supporting_facts": row.get("supporting_facts", {}),
        "prediction": prediction,
        "model": model,
        "retrieved": row["retrieved"],
        "prompt": row["prompt"],
    }
    if "initial_prediction" in row:
        output["initial_prediction"] = row["initial_prediction"]
    return output


def main():
    env_path = load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="results/hotpotqa_top5_prompts.jsonl")
    parser.add_argument("--out", default="results/hotpotqa_top5_answers.jsonl")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=int(os.getenv("LLM_MAX_TOKENS", "64")))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("LLM_TIMEOUT", "60")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("LLM_RETRIES", "3")))
    parser.add_argument("--retry_sleep", type=float, default=float(os.getenv("LLM_RETRY_SLEEP", "5")))
    args = parser.parse_args()

    client = get_client(args.timeout)
    if env_path.exists():
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"No .env file found at {env_path}; using shell environment variables.")
    rows = list(read_jsonl(args.prompts))
    if args.limit is not None:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    first_output = None
    if args.resume and out_path.exists():
        for existing in read_jsonl(out_path):
            completed_ids.add(existing["id"])
            if first_output is None:
                first_output = existing

    pending_rows = [row for row in rows if row["id"] not in completed_ids]
    mode = "a" if args.resume and out_path.exists() else "w"
    saved_count = len(completed_ids) if mode == "a" else 0

    with out_path.open(mode, encoding="utf-8") as f:
        for row in tqdm(pending_rows):
            response = create_completion_with_retries(client, args, row)
            prediction = response.choices[0].message.content.strip()
            output = build_output_row(row, prediction, args.model)
            if first_output is None:
                first_output = output
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            saved_count += 1
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Saved {saved_count} answers to {args.out}")
    if first_output:
        print("First prediction:")
        print(first_output["question"])
        print(f"gold: {first_output['gold_answer']}")
        print(f"pred: {first_output['prediction']}")


if __name__ == "__main__":
    main()
