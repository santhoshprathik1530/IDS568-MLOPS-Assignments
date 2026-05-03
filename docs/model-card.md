# Model Card

## Model Summary

This project uses a retrieval-augmented generation workflow rather than a single fine-tuned task model. The core response model is:

- `meta-llama/Llama-3.1-8B-Instruct`
- served through OpenRouter

The system also depends on:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- retriever: `FAISS IndexFlatL2`
- document corpus: 8 small course-focused knowledge documents embedded directly in `src/common/rag_pipeline.py`

## Intended Use

The intended use is a small, monitored RAG + agent service for answering questions and completing short analysis tasks about the course topics used in the project corpus.

This system is appropriate for:

- grounded question answering over the included knowledge base
- short structured evidence extraction
- simple agent tasks that depend on retrieval followed by one synthesis step

## Out-of-Scope Use

This system is not designed for:

- open-domain factual question answering
- legal, medical, or financial decision support
- autonomous multi-step tool use beyond the narrow course-task workflow
- handling regulated or sensitive personal data

## Training and Data Description

The generative model itself was not trained as part of this project. It is an external open-weight instruct model accessed through OpenRouter.

The local knowledge base used for retrieval consists of 8 short, synthetic course-aligned documents covering:

- RAG architecture
- chunking strategy
- embedding models
- FAISS and vector retrieval
- grounding and citations
- latency measurement
- agent tool-use policy
- failure analysis

Because the corpus is small and synthetic, this system should be understood as a course project artifact rather than a domain-complete knowledge assistant.

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

The biggest limitation is the size and scope of the local corpus. Retrieval quality is constrained by the fact that only 8 documents are available, so top-k retrieval can easily return a mix of useful and only partly relevant chunks.

Observed failure modes include:

- partial retrieval where only one of multiple expected supporting documents is returned
- latency spikes in generation
- occasional missing citations in final answers
- weak support for questions outside the narrow corpus boundary

The agent also depends on a constrained stopping policy. Without that policy, earlier versions of the controller looped too much and produced inefficient traces.

## Ethical Risks and Considerations

This system has relatively low direct social impact because the corpus is small and course-specific, but there are still governance concerns:

- answers may appear more confident than the retrieved evidence justifies
- missing citations can hide weak grounding
- stale or incomplete knowledge can lead to misleading summaries
- users might overgeneralize the system beyond its intended scope

## Monitoring and Governance Hooks

The final project adds operational controls around the base system:

- Prometheus metrics for latency, request counts, anomaly signals, drift score, and agent step counts
- A/B testing simulation to evaluate proposed changes before rollout
- Audit trail logging for version and intervention events
- Drift and risk analysis in later components of the final project

## Human Oversight

This system should be treated as an assistive tool rather than an authoritative one. For any use beyond the course demonstration setting, human review would be required whenever:

- citations are missing
- drift signals are elevated
- retrieval support appears weak
- the task affects a high-risk downstream decision
