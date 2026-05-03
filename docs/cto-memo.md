# Recommendations to a CTO

## Summary

I reviewed the final-project system as if it were a small production AI workflow. The system is a monitored RAG + agent service built on a local corpus, FAISS retrieval, and an 8B instruct model accessed through OpenRouter.

The good news is that the system is observable and reasonably disciplined. It has:

- request and latency monitoring
- drift and integrity analysis
- an audit trail
- a model card and risk register
- a bounded agent policy that avoids uncontrolled tool loops

## Key Findings

1. **The main bottleneck is generation latency, not retrieval**
   - Retrieval stays fast.
   - Generation time dominates end-user latency.

2. **The main quality risk is incomplete grounding rather than total failure**
   - The system usually retrieves something useful.
   - The bigger problem is partial retrieval, where the answer looks grounded but is narrower than it should be.

3. **Query complexity drift is a real operational concern**
   - The drift analysis showed a strong shift toward longer and more complex prompts.
   - That kind of drift is likely to reduce grounding quality before it breaks the service outright.

4. **The agent is reliable only because the controller is tightly constrained**
   - The agent works well in the current project because tool selection is stage-bounded.
   - If the workflow were widened without adding stronger safeguards, tool misuse risk would grow quickly.

5. **The external API boundary is acceptable for coursework but would be a major architecture question in production**
   - The current setup is fine for this assignment because the data is synthetic and the API path was explicitly approved.
   - It would need a much deeper privacy and compliance review in a real business setting.

## Recommended Actions

### High Priority

- Add production-style alerts for:
  - p95 answer latency
  - elevated drift score
  - repeated integrity anomalies
  - increased missing-citation rate

- Establish a corpus refresh policy:
  - define who owns corpus updates
  - define when re-indexing is required
  - log those events in the audit trail

### Medium Priority

- Treat long and multi-question prompts as a separate routing problem
  - they are the clearest drift signal in this project
  - they are also a likely cause of weaker grounding

- Run the A/B variant under live observation
  - the simulation supports shipping Variant B
  - the next step would be validating those gains against real monitored traffic

### Lower Priority

- Expand the corpus beyond the current 8-document set
  - this would make the system less brittle
  - it would also make the drift analysis more realistic

## Bottom Line

If I had to summarize the system in one sentence, I would say this:

> The project is operationally aware and well structured for a course environment, but its long-term reliability depends more on retrieval quality, corpus maintenance, and query-drift control than on the base model alone.

That means the next investments should go into observability, corpus governance, and query handling rather than chasing raw model size.
