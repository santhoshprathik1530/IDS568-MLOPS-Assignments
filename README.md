# Milestone 6: RAG Pipeline and Agent Workflow

This repository contains my submission for IDS 568 Milestone 6. The project has two parts:

1. a retrieval-augmented generation pipeline
2. a multi-tool agent that reuses the retriever as one of its tools

The original milestone instructions prefer local or self-hosted inference. For this submission, I used OpenRouter with professor approval, and both the RAG pipeline and the agent were evaluated with the same 8B instruct model.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Export the model configuration:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

## Model Used

- Model family: `meta-llama/Llama-3.1-8B-Instruct`
- OpenRouter slug: `meta-llama/llama-3.1-8b-instruct`
- Size class: `8B`
- Serving path: OpenRouter API

Both `rag_pipeline.py` and `agent_controller.py` use the same model path. By default they read `OPENROUTER_MODEL`, but the model can also be overridden with `--llm-model`.

## Usage

Run the RAG pipeline on one question:

```bash
python rag_pipeline.py --question "What is retrieval-augmented generation?"
```

Run the 10-query RAG evaluation:

```bash
python rag_pipeline.py --evaluate --export-json rag_eval_results.json
```

Run the agent on one task:

```bash
python agent_controller.py --task "Explain when an agent should retrieve before summarizing."
```

Run the 10-task agent evaluation:

```bash
python agent_controller.py --evaluate --trace-dir agent_traces
```

## Project Structure

- `rag_pipeline.py`: document chunking, embeddings, FAISS indexing, retrieval, grounded answer generation, and evaluation
- `agent_controller.py`: tool selection, retrieval integration, summarization/extraction tools, and trace export
- `rag_evaluation_report.md`: Part 1 analysis
- `agent_report.md`: Part 2 analysis
- `rag_pipeline_diagram.md`: pipeline diagram
- `agent_traces/`: exported trace files for the 10 agent tasks

## Knowledge Base

The RAG knowledge base is a small course-focused corpus defined directly in `rag_pipeline.py`. It contains these eight source documents:

- `rag_architecture.md`
- `chunking_strategy.md`
- `embedding_models.md`
- `faiss_notes.md`
- `grounding_practices.md`
- `latency_measurements.md`
- `agent_policy.md`
- `failure_analysis.md`

I used this smaller corpus to keep the retrieval behavior easy to inspect during evaluation.

## Architecture Overview

The RAG pipeline follows a standard flow:

1. split each source document into overlapping chunks
2. generate embeddings with `all-MiniLM-L6-v2`
3. store vectors in `FAISS IndexFlatL2`
4. retrieve the top-k chunks for a question
5. build a grounded prompt with source information
6. generate an answer with the instruct model

The agent uses that same retriever and adds two more tools:

- `summarize_context`
- `extract_evidence`

The final controller policy is intentionally simple: retrieve first, optionally use one synthesis tool, and then produce a final grounded answer. That policy gave much more reliable traces than letting the planner loop freely.

## Evaluation Setup

The exact commands used for the measured runs were:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
python rag_pipeline.py --evaluate --export-json rag_eval_results.json
python agent_controller.py --evaluate --trace-dir agent_traces
```

## Runtime Environment

- Operating system: macOS
- Architecture: Apple Silicon / arm64
- Python version: `3.12.12`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector index: `FAISS IndexFlatL2`
- Inference path: OpenRouter with `meta-llama/llama-3.1-8b-instruct`

## Observed Performance

From the measured evaluation runs:

- Average retrieval latency: `77.5 ms`
- Average RAG generation latency: `12202.6 ms`
- Average RAG end-to-end latency: `12280.2 ms`
- Agent success rate on 10 tasks: `1.00`
- Average agent task duration: `6698.6 ms`

## Known Limitations

- The corpus is small and synthetic, so retrieval precision is limited.
- Top-3 retrieval often includes extra context even when recall is good.
- API-based inference introduced some latency variance, including one major outlier in the RAG run.
- This project is designed to meet the milestone requirements, not to serve as a production-scale RAG system.
