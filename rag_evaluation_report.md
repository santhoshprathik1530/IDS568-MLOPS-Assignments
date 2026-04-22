# RAG Evaluation Report

## System Overview

The RAG system in `rag_pipeline.py` performs recursive chunking, sentence-transformer embeddings, FAISS exact retrieval, and grounded generation using a real instruct model accessed through OpenRouter.

## Design Decisions

- Chunking strategy: paragraph-first recursive chunking with overlap.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector index: `FAISS IndexFlatL2` for deterministic exact search on a small corpus.
- Generator: OpenRouter-served 7B-14B instruct model such as `meta-llama/llama-3.1-8b-instruct`.
- Corpus: 8 small MLOps course-focused source documents embedded directly in `rag_pipeline.py`.

## Retrieval Accuracy on 10 Handcrafted Queries

Run:

```bash
python rag_pipeline.py --evaluate --export-json rag_eval_results.json
```

Measured evaluation file: `rag_eval_results.json`

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

Average metrics:

- Precision@k: `0.367`
- Recall@k: `0.85`
- Hit rate: `1.00`
- Retrieval latency: `77.5 ms`
- Generation latency: `12202.6 ms`
- End-to-end latency: `12280.2 ms`

## Qualitative Grounding Analysis

- Strong grounding cases appeared on chunking, embeddings, and agent-policy questions, where the model used the retrieved context directly and produced concise source-grounded explanations.
- Query `q5` performed best on precision because both `grounding_practices.md` and `rag_architecture.md` were retrieved and directly matched the answer requirements.
- The weakest cases were `q1`, `q8`, and `q9`, where only one of the two relevant documents appeared in the top-3 results. These are partial retrieval cases rather than complete misses.
- The answers generally stayed on-topic because the prompt explicitly instructed the model to answer only from provided context and cite sources inline.

## Error Attribution

- Retrieval failures:
  - The main issue was low precision caused by retrieving 3 chunks from a very small 8-document corpus. Relevant information was usually present, but extra non-essential chunks were often returned.
  - Queries `q1`, `q8`, and `q9` show recall loss because only one of the expected supporting sources appeared in the retrieved set.
- Generation or grounding failures:
  - No complete hallucination was observed in the measured run, but the quality varied with retrieval quality. When retrieval returned only partial evidence, the model produced narrower answers.
  - Generation latency fluctuated heavily, which affects usability even when answer quality is acceptable.
- Model-capacity limitations:
  - Query `q7` had an unusually high generation latency of about `82.4s`, showing that API-backed hosted inference can introduce large tail latency even for otherwise straightforward prompts.
  - With a small synthetic corpus, the model had limited opportunity to demonstrate deeper multi-hop synthesis.

## Latency Analysis

- Retrieval latency:
  - Retrieval was consistently fast, averaging `77.5 ms`, with a minimum of `23.5 ms` and a maximum of `187.1 ms`.
- Generation latency:
  - Generation dominated runtime, averaging `12202.6 ms`. Most responses finished in a few seconds, but there was one major outlier at `82371.9 ms`.
- End-to-end latency:
  - End-to-end latency closely tracked generation latency, averaging `12280.2 ms`, which confirms that retrieval is not the bottleneck in this implementation.

## Limitations

- The included corpus is intentionally small for milestone-scale testing.
- The evaluation used an instructor-approved OpenRouter API path rather than local serving.
- Precision remains limited because the corpus is small and top-3 retrieval often includes extra chunks.
