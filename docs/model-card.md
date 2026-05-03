# Model Card

## Model Summary

This project does not revolve around a single fine-tuned task model. Instead, it uses a retrieval-augmented generation workflow with a lightweight agent layer. The main generation model is:

- `meta-llama/Llama-3.1-8B-Instruct`
- accessed through OpenRouter

The surrounding retrieval stack includes:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- vector store: `FAISS IndexFlatL2`
- document corpus: eight short course-focused documents defined in `src/common/rag_pipeline.py`

## Intended Use

The intended use is a small RAG plus agent service for grounded question answering and short evidence-based tasks over the included project corpus.

In its current form, the system is appropriate for:

- answering course-related questions with retrieved support
- short evidence extraction tasks
- simple agent workflows that rely on retrieval followed by one synthesis step

## Out-of-Scope Use

This system is not intended for:

- open-domain factual assistance
- legal, financial, or medical decision support
- autonomous multi-step tool execution outside the narrow project workflow
- handling sensitive or regulated personal data

## Training and Data Description

The generation model itself was not trained as part of this project. It is an externally hosted open-weight instruct model.

The local retrieval corpus is intentionally small and synthetic. It contains eight short documents covering:

- RAG architecture
- chunking strategy
- embedding models
- FAISS retrieval
- grounding and citation practice
- latency measurement
- agent tool-use policy
- failure analysis

Because the corpus is narrow and course-specific, this system should be treated as a project artifact rather than as a general-purpose knowledge assistant.

## Performance

### RAG Evaluation

From the 10-query evaluation run:

- average precision@k: `0.367`
- average recall@k: `0.85`
- average hit rate: `1.00`
- average retrieval latency: `77.5 ms`
- average generation latency: `12202.6 ms`
- average end-to-end latency: `12280.2 ms`

### Monitoring Run

From the monitoring simulation:

- average retrieval latency: `56.02 ms`
- average generation latency: `2420.03 ms`
- maximum generation latency: `4970.4 ms`
- requests without source citations: `3`

### Agent Evaluation

From the 10-task agent evaluation:

- success rate: `1.00`
- average task duration: `6698.6 ms`
- average step count: `1.9`

## Limitations and Failure Modes

The most important limitation is the size of the corpus. With only eight documents, retrieval quality is bounded by how much of the answer can be supported by a very small knowledge base.

Observed failure modes include:

- partial retrieval, where only part of the relevant support is returned
- latency spikes during generation
- occasional missing citations in the final answer
- weak performance on questions that stretch beyond the project corpus

The agent also depends on a constrained stopping policy. Earlier versions of the controller were more likely to overuse tools, which made traces longer and less reliable.

## Ethical Risks and Considerations

The project has relatively low direct social impact because it is built around a synthetic, course-specific corpus. Even so, there are meaningful governance concerns:

- the model may sound more confident than the evidence supports
- missing citations can hide weak grounding
- stale or incomplete knowledge can lead to misleading summaries
- users may overextend the system beyond its intended scope

## Monitoring and Governance Hooks

The final project adds several operational controls around the base system:

- Prometheus metrics for latency, request count, anomaly signals, drift score, and agent step counts
- an A/B simulation for evaluating retrieval and formatting changes
- a structured audit trail for model and intervention events
- drift analysis and formal risk review documents

## Human Oversight

This system should be treated as an assistive tool, not as an authority. Human review is especially important when:

- citations are missing
- drift signals rise above baseline
- retrieval support appears weak
- a task could affect a high-stakes decision
