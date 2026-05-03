# Governance Review

## System Boundary

The system reviewed here is a small RAG plus agent workflow built from a local document corpus, sentence-transformer embeddings, FAISS retrieval, and an instruct LLM accessed through OpenRouter. The agent layer is intentionally narrow and is limited to retrieval followed by at most one synthesis step.

For the purposes of this review, the system boundary begins when a user submits a question or task and ends when the final response is returned. Inside that boundary, the system performs integrity checks, retrieval, prompt construction, model inference, optional tool-assisted synthesis, and metric emission.

## Data Security

This project does not use internal enterprise data or real sensitive records, which keeps the immediate security risk fairly low. Even so, the inference boundary still matters because prompts are sent to an external API provider. If a user were to enter private or regulated information, that content would leave the local environment.

In the current class setting, the most reasonable mitigation is procedural:

- keep the workload limited to non-sensitive course-style content
- document the external inference path clearly
- avoid using the system for real personal or regulated data

If this design were moved beyond the course setting, the external API boundary would require a more formal privacy and vendor review.

## Retrieval Risks

The most important retrieval risks in this project are stale knowledge, partial retrieval, and contaminated context.

The corpus is static and manually maintained, so stale knowledge is a realistic concern. Even if the software remains unchanged, retrieval quality can decline if the corpus no longer matches the request distribution.

Partial retrieval is another practical issue. Because the corpus is small and the system uses top-k retrieval, it is possible to retrieve something relevant without retrieving everything needed for a complete answer. That pattern already appeared in the earlier milestone evaluation.

The third retrieval risk is low-quality or contaminated context. If a source document were misleading, outdated, or adversarial, the language model could still produce a confident answer grounded in poor evidence.

Monitoring and drift analysis help reveal some of these problems, but they do not solve them automatically. They improve observability; they do not guarantee correctness.

## Hallucination Risk Points

In this system, the main hallucination risk does not come from answering without retrieval. It comes from answering with retrieval that is incomplete.

The design reduces hallucination risk in three ways:

- retrieval is required before answering
- the prompt encourages grounded output with citations
- monitoring tracks citation gaps and weak-support signals

Even with those controls, hallucination can still appear when only part of the relevant context is retrieved, when the prompt contains multiple sub-questions, or when the model summarizes beyond what the evidence cleanly supports. In other words, grounded output is safer than ungrounded output, but it is not automatically correct.

## Tool-Misuse Pathways

The agent in this project is deliberately constrained, but that does not eliminate tool risk entirely. The main failure modes are:

- selecting a synthesis action too early
- summarizing incomplete evidence as if it were complete
- extracting facts from a partially relevant retrieval set

The current controller mitigates this by forcing a simple action order: retrieve first, allow at most one synthesis step, and then return a final answer. This makes the agent less flexible, but it also makes the system easier to audit and reason about.

## Compliance Concerns

The main compliance concerns are tied to data handling and scope. A user could submit content that should not be sent to an external inference provider. A second concern is expectation management: because the system is narrow and course-specific, it would be inappropriate to present it as a general-purpose assistant.

Within the class setting, the setup is acceptable because:

- the corpus is synthetic and non-sensitive
- the model path is documented
- API-based inference was explicitly approved

Outside the class setting, the same design would need stronger controls around privacy, retention, access policy, and vendor governance.

## Overall Governance Assessment

My overall assessment is that the system is governance-aware but intentionally lightweight. Its strength is not that it removes risk altogether. Its strength is that it makes key risks visible through monitoring, documentation, and constrained workflow design.

The main remaining weakness is that response quality still depends heavily on corpus quality and on the complexity of incoming queries. That is why the retrieval-risk and drift stories are central to this project.
