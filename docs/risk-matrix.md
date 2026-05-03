# Risk Matrix

| Risk | Likelihood | Severity | Why It Matters | Mitigation |
|---|---|---|---|---|
| Partial retrieval returns incomplete evidence | High | Medium | The answer may sound grounded but still miss part of the needed context. | Track retrieval confidence, require citations, and review low-support responses. |
| Generation latency spikes | Medium | Medium | Slow responses reduce usability and can hide queueing or provider-side instability. | Monitor p95 latency, separate retrieval and generation metrics, and alert on sustained degradation. |
| Missing citations in answers | Medium | Medium | Missing citations weaken transparency and make human review harder. | Monitor citation failures and prefer citation-focused prompt variants. |
| Query complexity drift | Medium | Medium | Longer and multi-question prompts can degrade retrieval quality and grounding. | Track token-count drift, integrity anomalies, and review drift windows regularly. |
| Stale corpus | High | Medium | The system may retrieve outdated or incomplete knowledge even when retrieval appears to work. | Define a corpus refresh and re-index process and record updates in the audit trail. |
| Sensitive content sent to external API | Medium | High | Prompt content could be exposed outside the local environment. | Restrict the system to non-sensitive inputs and document the inference boundary clearly. |
| Agent uses the wrong synthesis step | Medium | Medium | A wrong intermediate action can compress incomplete evidence into a misleading answer. | Keep the tool set narrow and constrain action order by stage. |
| Prompt-like or malformed inputs | Medium | Medium | Unusual inputs can stress the service or reduce grounding quality. | Track input anomalies and apply integrity screening before generation. |
| User over-trust in a narrow system | Medium | High | Users may treat a course-specific system as if it were domain general. | Document intended use, out-of-scope use, and require human review for high-stakes contexts. |
| Policy or compliance mismatch from API-based inference | Low | High | A setup acceptable for coursework may not satisfy real deployment constraints. | Record the approved exception and avoid presenting the architecture as production-ready by default. |
