# Advanced RAG Evaluation (LlamaIndex + RAGAS)
# Javier Castro dnAI 2025

This notebook runs a compact RAG evaluation loop with multiple retrieval strategies, optional reranking, and RAGAS scoring to compare retrieval/answer quality.

## What it does
- Bootstraps deps on the fly: `llama-index`, `ragas`, `openai`, `pandas`, `tabulate`, optional `cohere`.
- Builds a corpus from `./data` (auto-creates a small sample doc if empty).
- Runs several experiments:
  - `baseline-vector` (vanilla vector search)
  - `sentence-window` (sentence-window chunks)
  - `auto-merging` (hierarchical auto-merge retriever)
  - `sentence-window+rerank` (with reranker: cohere or LLM/embedding fallback)
  - `auto-merging+rerank` (same rerank options)
- Answers extractively using OpenAI, then scores with RAGAS metrics (answer relevance, faithfulness/groundedness, context recall, context precision).
- Prints per-question traces and Markdown tables, and saves CSV artifacts under `./outputs`.

## Requirements
- Python 3.10+.
- Environment variables:
  - `OPENAI_API_KEY` (required).
  - `COHERE_API_KEY` (optional; used only if you select the Cohere reranker).
- A writable `./outputs` directory.
- `./data` folder with your corpus (any text files). If empty, a tiny sample file is auto-created.

## Quick start
1) Set env vars, e.g. `export OPENAI_API_KEY=... [COHERE_API_KEY=...]`.
2) Put your documents in `./data` (or leave empty to use the sample).
3) Open and run all cells in `Advanced RAG.ipynb` (or run via `jupyter nbconvert --execute`).
4) Inspect results in the notebook output and in `./outputs`:
   - `traces_all.csv` (all Q/A/context rows)
   - `traces/<experiment>.csv` (per-experiment traces)
   - `ragas_raw_scores.csv` (per-sample metrics)
   - `ragas_leaderboard.csv` (mean metrics per experiment, if available)

## Configuration (edit `CONFIG` at the top of the notebook)
- `data_path`: corpus folder (default `./data`).
- `eval_questions` / `eval_references`: evaluation prompts and optional gold references; if references are missing, the notebook auto-generates concise references from the corpus.
- `embedding`, `llm`, `ragas`: model names (OpenAI) for retrieval and scoring.
- `experiments`: list of retriever/reranker configs (top-k, window size, reranker type/model/top_n).
- `output_dir`: where CSVs are written.
- `print_context_chars`: set >0 to print context snippets.

## Notes & troubleshooting
- If Cohere rerank is configured without `COHERE_API_KEY` or the package, the notebook falls back to an LLM/embedding reranker and prints a warning.
- Long runs may consume OpenAI tokens; adjust `eval_questions`, `experiments`, or top-k values to control cost.
- If you see “Set OPENAI_API_KEY”, export the key and rerun.
