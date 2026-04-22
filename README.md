# Milestone 6: RAG Pipeline and Multi-Tool Agent

This repository contains a local Retrieval-Augmented Generation (RAG) pipeline and a multi-tool agent built for IDS 568 Milestone 6.

Instructor accommodation note: the original milestone prefers local or self-hosted inference, but the professor approved API-based evaluation for this submission. This implementation therefore uses OpenRouter for the LLM calls.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Export your OpenRouter credentials:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

If your professor approved API-based evaluation, this repository uses OpenRouter for generation. You can override the model at runtime with `--llm-model`.

## Model Configuration

This project was evaluated with the following model configuration:

- Model family: `meta-llama/Llama-3.1-8B-Instruct`
- OpenRouter model slug: `meta-llama/llama-3.1-8b-instruct`
- Size class: `8B`
- Access path: OpenRouter API
- Accommodation note: API-based inference was used with professor approval

Both `rag_pipeline.py` and `agent_controller.py` use the same model path. By default they read `OPENROUTER_MODEL`, and you can override that with `--llm-model` on either script.

## Usage

Run the RAG pipeline on a single question:

```bash
python rag_pipeline.py --question "What is retrieval-augmented generation?"
```

Run the 10-query RAG evaluation set:

```bash
python rag_pipeline.py --evaluate --export-json rag_eval_results.json
```

Run the agent on one task:

```bash
python agent_controller.py --task "Explain when an agent should retrieve before summarizing."
```

Run the 10-task agent evaluation set and export observable traces:

```bash
python agent_controller.py --evaluate --trace-dir agent_traces
```

## Architecture Overview

- `rag_pipeline.py`: chunking, embeddings, FAISS indexing, retrieval, grounded generation, evaluation.
- `agent_controller.py`: LLM-driven tool selection with retrieval, summarization, extraction, and exported traces.
- `agent_traces/`: generated JSON traces for the 10 evaluation tasks.

## Knowledge Base

The RAG knowledge base is a small in-repo corpus defined directly in `rag_pipeline.py`. It contains these eight source documents:

- `rag_architecture.md`
- `chunking_strategy.md`
- `embedding_models.md`
- `faiss_notes.md`
- `grounding_practices.md`
- `latency_measurements.md`
- `agent_policy.md`
- `failure_analysis.md`

These documents cover the exact course concepts the milestone asks for: RAG architecture, chunking, embeddings, vector indexing, grounding, latency measurement, agent tool policy, and failure analysis.

## Model Serving

- Recommended final model: `meta-llama/llama-3.1-8b-instruct`
- Size class: 8B
- Serving stack: OpenRouter API
- Runtime target for final evaluation: OpenRouter-backed hosted inference with instructor approval
- Environment variable: `OPENROUTER_API_KEY`
- Optional model override: `OPENROUTER_MODEL` or `--llm-model`
- Runtime used for this evaluation: macOS arm64, Python `3.12.12`
- Typical retrieval latency observed: about `77.5 ms`
- Typical generation latency observed: about `12.2 s` on average, with one large tail-latency outlier

## Serving and Invocation

This repository does not start a local model server. Instead, both scripts send generation requests directly to OpenRouter using the configured API key and model slug.

The exact invocation pattern used for evaluation was:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
python rag_pipeline.py --evaluate --export-json rag_eval_results.json
python agent_controller.py --evaluate --trace-dir agent_traces
```

This satisfies the README requirement to document the exact serving/inference path used for the final evaluated runs.

## Runtime Environment

- Operating system: macOS
- Architecture: arm64 / Apple Silicon
- Python version: `3.12.12`
- Embedding stack: `sentence-transformers` with `all-MiniLM-L6-v2`
- Retrieval index: `FAISS IndexFlatL2`
- LLM inference path: OpenRouter-hosted `meta-llama/llama-3.1-8b-instruct`

## Submission Status

The repository currently contains the required code, reports, diagram, requirements file, README, RAG evaluation JSON, and 10 agent trace files.

Two submission items still depend on how you package the final repository:

- Repository naming: the course asks for `ids568-milestone6-[your_netid]`
- Git tagging and push: the current working directory is not a git repository, so `git tag submission && git push --tags` still has to be done after you create or connect the final repo

## Known Limitations

- Final graded runs still require a real 7B-14B instruct model to be configured on OpenRouter.
- The included corpus is small and intended as a milestone-sized demonstration corpus.
- Retrieval precision is modest because the evaluation uses only 8 short documents and top-3 retrieval.
- The corpus is synthetic course-focused material rather than a larger external document collection.
