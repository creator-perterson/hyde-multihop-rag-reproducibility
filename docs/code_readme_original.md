# Agentic RAG Experiments

## Step 1: Prepare HotpotQA Sample

Activate your conda environment first:

```powershell
conda activate agentic_rag
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare a small HotpotQA sample:

```powershell
python .\src\data\prepare_hotpotqa_sample.py --limit 200 --streaming
```

Inspect the saved files:

```powershell
python .\src\data\inspect_hotpotqa_sample.py
```

Generated files:

```text
datasets/hotpotqa_sample/questions.jsonl
datasets/hotpotqa_sample/corpus.jsonl
```

## Step 2: Build a FAISS Index

```powershell
python .\src\retriever\build_faiss_index.py
```

Generated files:

```text
datasets/hotpotqa_sample/faiss_index/index.faiss
datasets/hotpotqa_sample/faiss_index/docstore.jsonl
datasets/hotpotqa_sample/faiss_index/metadata.json
```

## Step 3: Retrieve Top-k Evidence

```powershell
python .\src\retriever\retrieve_topk.py --top_k 5
```

Generated file:

```text
results/hotpotqa_top5_retrieval.jsonl
```

## Step 4: Evaluate Retrieval

```powershell
python .\src\evaluation\evaluate_retrieval.py
```

This checks whether the retrieved top-5 documents contain the gold supporting titles from HotpotQA.

## Network-Free Backup: TF-IDF Retrieval

If Hugging Face model download fails, run this first. It does not need an embedding model:

```powershell
python .\src\retriever\retrieve_tfidf_topk.py --top_k 5
python .\src\evaluation\evaluate_retrieval.py --retrieval results/hotpotqa_top5_tfidf_retrieval.jsonl
```

TF-IDF is weaker than dense embeddings, but it is a valid lightweight retrieval baseline and is useful for making sure the whole RAG pipeline works.

## Step 5: Build RAG Prompts

```powershell
python .\src\generator\build_rag_prompts.py --retrieval results/hotpotqa_top5_retrieval.jsonl --style infer
```

Generated file:

```text
results/hotpotqa_top5_prompts.jsonl
```

## Step 6: Generate Answers with an OpenAI-Compatible LLM API

Install the OpenAI client if needed:

```powershell
pip install openai python-dotenv
```

Recommended: create a `.env` file under the repository root.

For OpenAI:

```env
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
```

For other OpenAI-compatible providers:

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-provider-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

Then run:

```powershell
python .\src\generator\generate_answers_openai_compatible.py --limit 10 --model gpt-4o-mini
```

You can also set keys in the current PowerShell session instead of `.env`:

```powershell
$env:LLM_API_KEY="your_api_key_here"
$env:LLM_BASE_URL="https://your-provider-compatible-endpoint/v1"
$env:LLM_MODEL="your-model-name"
python .\src\generator\generate_answers_openai_compatible.py --limit 10
```

Start with `--limit 10` to avoid spending too much money while testing.

## Step 7: Evaluate Generated Answers

```powershell
python .\src\evaluation\evaluate_answers.py --answers results/hotpotqa_top5_answers.jsonl
```

Analyze wrong examples:

```powershell
python .\src\evaluation\analyze_answer_errors.py --answers results/hotpotqa_top5_answers.jsonl
```

## Step 8: Try Larger Top-k

If generation errors are caused by missing evidence, retrieve more documents:

```powershell
python .\src\retriever\retrieve_topk.py --top_k 10 --out results/hotpotqa_top10_retrieval.jsonl
python .\src\evaluation\evaluate_retrieval.py --retrieval results/hotpotqa_top10_retrieval.jsonl
python .\src\generator\build_rag_prompts.py --retrieval results/hotpotqa_top10_retrieval.jsonl --out results/hotpotqa_top10_prompts.jsonl --top_k 10 --style infer
python .\src\generator\generate_answers_openai_compatible.py --prompts results/hotpotqa_top10_prompts.jsonl --out results/hotpotqa_top10_answers.jsonl --limit 10
python .\src\evaluation\evaluate_answers.py --answers results/hotpotqa_top10_answers.jsonl
python .\src\evaluation\analyze_answer_errors.py --answers results/hotpotqa_top10_answers.jsonl
```

## Step 9: Try Iterative Second-Hop Retrieval

If top-10 still misses second-hop evidence, run an iterative retrieval baseline:

```powershell
python .\src\retriever\retrieve_iterative_topk.py --first_retrieval results/hotpotqa_top10_retrieval.jsonl --top_k 10 --out results/hotpotqa_iterative_top10_retrieval.jsonl
python .\src\evaluation\evaluate_retrieval.py --retrieval results/hotpotqa_iterative_top10_retrieval.jsonl
python .\src\generator\build_rag_prompts.py --retrieval results/hotpotqa_iterative_top10_retrieval.jsonl --out results/hotpotqa_iterative_top10_prompts.jsonl --top_k 10 --style infer
python .\src\generator\generate_answers_openai_compatible.py --prompts results/hotpotqa_iterative_top10_prompts.jsonl --out results/hotpotqa_iterative_top10_answers.jsonl --limit 10
python .\src\evaluation\evaluate_answers.py --answers results/hotpotqa_iterative_top10_answers.jsonl
python .\src\evaluation\analyze_answer_errors.py --answers results/hotpotqa_iterative_top10_answers.jsonl
```

This is a simple Agentic RAG-style idea: first retrieve likely evidence, then use that evidence to guide a second retrieval round.

If the iterative retrieval finds the right evidence but the model gives a vague or incomplete answer, use the extractive prompt:

```powershell
python .\src\generator\build_rag_prompts.py --retrieval results/hotpotqa_iterative_top10_retrieval.jsonl --out results/hotpotqa_iterative_top10_extractive_prompts.jsonl --top_k 10 --style extractive
python .\src\generator\generate_answers_openai_compatible.py --prompts results/hotpotqa_iterative_top10_extractive_prompts.jsonl --out results/hotpotqa_iterative_top10_extractive_answers.jsonl --limit 10
python .\src\evaluation\evaluate_answers.py --answers results/hotpotqa_iterative_top10_extractive_answers.jsonl
python .\src\evaluation\analyze_answer_errors.py --answers results/hotpotqa_iterative_top10_extractive_answers.jsonl
```
