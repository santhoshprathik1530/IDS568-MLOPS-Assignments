# Recommendations to a CTO

## Executive Summary

I reviewed this project as though it were a small production-facing AI workflow rather than only a class exercise. The system is a monitored RAG plus agent service built on a local corpus, FAISS retrieval, and an 8B instruct model accessed through OpenRouter.

Overall, the system is better instrumented and better documented than a typical prototype. It includes monitoring, drift analysis, an audit trail, a model card, and a constrained agent policy. At the same time, its long-term reliability depends less on the base model itself and more on retrieval quality, corpus maintenance, and how the system responds to changes in user query behavior.

## Key Findings

The first finding is that generation latency is the main performance bottleneck. Retrieval is relatively fast, but model inference dominates end-to-end response time.

The second finding is that the main quality risk is incomplete grounding rather than total failure. In most cases the system retrieves something relevant, but it can still return an answer that is narrower than the question warrants if the retrieval set is only partially complete.

The third finding is that query complexity drift is already visible in the synthetic production-style window. Longer and more compound prompts are likely to reduce grounding quality before they produce obvious system failures.

The fourth finding is that the agent behaves reliably mainly because it is tightly constrained. If the action space were expanded without stronger controls, tool misuse risk would increase quickly.

The fifth finding is that the external API boundary is acceptable for the class setting but would require a more serious privacy and compliance review in a real deployment.

## Recommended Actions

### High Priority

- add operational alerts for p95 latency, elevated drift score, repeated integrity anomalies, and citation failures
- define a corpus refresh and re-indexing policy with clear ownership
- record refresh and intervention events consistently in the audit trail

### Medium Priority

- treat long and multi-question prompts as a separate routing or preprocessing problem
- validate the A/B recommendation under live monitored traffic rather than relying only on simulation

### Lower Priority

- expand the corpus beyond the current eight-document set
- strengthen retrieval evaluation under more varied query types

## Closing Assessment

If I had to summarize the system in one sentence, I would describe it as operationally aware and well structured for a course environment, but still dependent on careful retrieval management and input monitoring for reliable long-term behavior.

For that reason, the next improvements should focus on observability, corpus governance, and query handling rather than on increasing model size.
