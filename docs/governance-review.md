# Governance Review

## System Boundary

The system under review is a small RAG + agent workflow built on top of:

- a local document corpus
- a sentence-transformer embedding model
- a FAISS retriever
- an instruct LLM accessed through OpenRouter
- two lightweight agent tools: summarization and evidence extraction

The system boundary starts when a user submits a question or task and ends when the final answer is returned. Within that boundary, the system performs integrity checks, retrieval, prompt construction, model inference, optional tool-assisted synthesis, and monitoring.

## Data Security

The project does not use proprietary or internal enterprise data, which reduces the immediate security risk. However, the inference path still matters because prompts are sent to an external API provider. That means a user could unintentionally transmit sensitive content if the service were used carelessly.

In the current project setup, the right mitigation is procedural rather than architectural:

- use only non-sensitive course-style content
- document the external API dependency clearly
- avoid real personal or regulated data

If this were moved toward a more serious deployment, the API boundary would need a much stricter review.

## Retrieval Risks

There are three main retrieval risks in this system:

1. **stale knowledge**
   - the corpus is static and manually maintained
   - if the content stops matching the query distribution, retrieval quality will degrade

2. **partial retrieval**
   - with a small corpus and top-k retrieval, the system can return some relevant context without returning all the context needed for a complete answer
   - this was already visible in the Milestone 6 evaluation

3. **contaminated or low-quality context**
   - if a document were poorly written or adversarial, the LLM could be grounded in bad evidence rather than good evidence

The monitoring and drift components help with some of this, but not all of it. Monitoring can reveal weaker grounding signals or changing query patterns. It cannot by itself guarantee that the underlying corpus remains correct.

## Hallucination Risk Points

The main hallucination risk does not come from retrieval being absent; it comes from retrieval being incomplete.

This project reduces hallucination risk in three ways:

- it forces retrieval before answering
- it asks for source-backed grounding
- it monitors signals such as missing citations and weak-support behavior

Even so, hallucination can still happen when:

- only one of several relevant chunks is retrieved
- the query contains multiple sub-questions
- the model summarizes beyond what the evidence cleanly supports

So the main governance position here is that “grounded” does not mean “safe by default.” It means the system is easier to inspect and easier to challenge.

## Tool-Misuse Pathways

The agent is intentionally narrow, but tool risk still exists.

The key misuse pathways are:

- selecting a synthesis tool before enough evidence is gathered
- summarizing an incomplete context as if it were complete
- extracting facts from a retrieval set that is only partially relevant

The current controller mitigates this by constraining the action flow: retrieve first, then at most one synthesis step, then finish. That makes the system less flexible but much easier to reason about and audit.

## Compliance Concerns

The main compliance concerns are:

- accidental submission of sensitive prompts to an external inference provider
- unclear user expectations around the limits of the corpus
- reuse of this course project beyond the low-risk context it was designed for

This project is acceptable within the class setting because:

- the corpus is synthetic and non-sensitive
- the model path is documented
- the API exception was explicitly approved

Outside that setting, the same design would require stronger controls around privacy, retention, access policy, and vendor review.

## Overall Governance Assessment

This system is governance-aware but still lightweight. Its strongest property is not that it eliminates risk, but that it makes risk more observable:

- monitoring surfaces latency, anomalies, and drift
- the audit trail records system events
- the model card and risk register document the boundaries clearly

The main remaining weakness is that correctness still depends heavily on corpus quality and query complexity. That is why the drift and retrieval-risk story is central to this project.
