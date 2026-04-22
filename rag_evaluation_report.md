# RAG Evaluation Report

## Overview

For Part 1, I built a small RAG pipeline in `rag_pipeline.py`. The pipeline chunks documents, generates embeddings, stores them in FAISS, retrieves the most relevant chunks for a question, and then sends a grounded prompt to the instruct model for answer generation.

The evaluation was run with:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- vector index: `FAISS IndexFlatL2`
- generation model: `meta-llama/llama-3.1-8b-instruct` through OpenRouter

The document corpus is small by design and is embedded directly in the code. It contains 8 short documents focused on the main Module 7 topics: RAG architecture, chunking, embeddings, vector indexing, grounding, latency, agent tool policy, and failure analysis.

## Design Decisions

I used paragraph-first recursive chunking with overlap because it keeps related ideas together while still allowing reasonably fine-grained retrieval. When a paragraph is too long, the code falls back to sentence-level splitting and then to a sliding window only when needed.

For embeddings, I used `all-MiniLM-L6-v2` because it is lightweight, easy to run, and commonly used for small semantic search experiments. For indexing, I used `FAISS IndexFlatL2` because the corpus is small and exact search makes retrieval behavior easier to inspect during evaluation.

## Retrieval Accuracy on 10 Handcrafted Queries

The evaluation results were exported to `rag_eval_results.json`.

| Query ID | Precision@k | Recall@k | Hit Rate | Retrieval Latency (ms) | Generation Latency (ms) | End-to-End (ms) |
|---|---:|---:|---:|---:|---:|---:|
| q1 | 0.33 | 0.50 | 1.00 | 49.8 | 2274.0 | 2323.9 |
| q2 | 0.33 | 1.00 | 1.00 | 81.6 | 690.0 | 771.7 |
| q3 | 0.33 | 1.00 | 1.00 | 54.5 | 19371.6 | 19426.2 |
| q4 | 0.33 | 1.00 | 1.00 | 128.2 | 9298.7 | 9427.0 |
| q5 | 0.67 | 1.00 | 1.00 | 69.8 | 3228.2 | 3298.1 |
| q6 | 0.33 | 1.00 | 1.00 | 78.5 | 370.4 | 448.9 |
| q7 | 0.33 | 1.00 | 1.00 | 24.4 | 82371.9 | 82396.4 |
| q8 | 0.33 | 0.50 | 1.00 | 187.1 | 2016.6 | 2203.8 |
| q9 | 0.33 | 0.50 | 1.00 | 77.8 | 587.5 | 665.4 |
| q10 | 0.33 | 1.00 | 1.00 | 23.5 | 1816.9 | 1840.5 |

Average values across the 10 queries:

- Precision@k: `0.367`
- Recall@k: `0.85`
- Hit rate: `1.00`
- Retrieval latency: `77.5 ms`
- Generation latency: `12202.6 ms`
- End-to-end latency: `12280.2 ms`

## Grounding Analysis

The answers were usually well grounded when the correct documents appeared in the top-3 retrieved results. Questions about chunking, embeddings, and agent policy were especially stable because the relevant evidence was usually retrieved cleanly and the prompt explicitly told the model to stay within the provided context.

The best example was `q5`, which asked about grounded versus hallucinatory answers. In that case both `grounding_practices.md` and `rag_architecture.md` were retrieved, so the model had direct support for the answer.

The weakest cases were `q1`, `q8`, and `q9`. In those cases the retriever still found relevant material, but only one of the expected supporting sources appeared in the retrieved set. The answers stayed on-topic, but they were narrower than they would have been with fuller evidence.

## Error Attribution

Most of the observed weakness came from retrieval, not from blatant hallucination. The main problem was low precision: with a small corpus and top-3 retrieval, the system often returned one or two useful chunks plus an extra chunk that was not central to the question.

There were also a few partial-recall cases. Queries `q1`, `q8`, and `q9` each recovered some of the needed material, but not all of it. Those are better described as incomplete retrieval cases than generation failures.

On the generation side, I did not observe a fully fabricated answer in the measured runs. The more noticeable issue was latency instability. One response took more than 80 seconds, which did not hurt correctness directly but would matter in a real deployment.

## Latency Discussion

Retrieval was consistently fast. Across the 10 queries, the average retrieval time was `77.5 ms`, with a minimum of `23.5 ms` and a maximum of `187.1 ms`. That confirms that embedding the query and searching the FAISS index are not the bottleneck here.

Generation time dominated the full pipeline. The average generation time was `12202.6 ms`, and end-to-end time was almost identical at `12280.2 ms`. Most generations finished in a few seconds, but `q7` produced a major outlier at `82371.9 ms`, which strongly affected the average.

## Limitations

This evaluation was done on a deliberately small corpus, so the measured precision should not be interpreted as a ceiling for the approach itself. A larger and more realistic document set would likely improve retrieval quality and make the evaluation more representative.

This submission also uses an instructor-approved OpenRouter API path rather than a local model server. That made the project easier to run, but it also introduced latency variability that would not necessarily appear in the same way on a dedicated local or self-hosted setup.
