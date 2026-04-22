# RAG Pipeline Diagram

```text
                +----------------------+
                |   Source Documents    |
                |  (local sample corpus)|
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Recursive Chunker     |
                | size + overlap policy |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Embedding Model       |
                | all-MiniLM-L6-v2      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | FAISS IndexFlatL2     |
                | vector store          |
                +----------+-----------+
                           |
         user query        | top-k chunks
     +----------------+    v
     | Query Embedder |--> Retriever
     +----------------+        |
                               v
                +------------------------------+
                | Prompt Builder                |
                | context + citations + policy |
                +--------------+---------------+
                               |
                               v
                +------------------------------+
                | Instruct Model Endpoint      |
                | OpenRouter-hosted 8B model   |
                +--------------+---------------+
                               |
                               v
                +------------------------------+
                | Grounded Answer + Citations  |
                | + latency measurements       |
                +------------------------------+
```

## Decision Points

- Chunking chooses paragraph-first recursive splitting with overlap.
- Retrieval selects the top-k chunks from the FAISS index.
- Generation is instructed to refuse unsupported claims and cite sources.
- Evaluation separates retrieval metrics from generation latency and end-to-end latency.
