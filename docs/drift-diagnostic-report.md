# Drift Diagnostic Report

## Overview

For this component, I treated drift as a change in the request distribution entering the RAG + agent system rather than a classic label-shift problem on a supervised dataset. That choice fits the architecture of this project better because the most important production question is whether the user-query mix is changing in ways that will degrade retrieval quality, grounding, or tool selection.

The drift analysis was run with:

- script: `src/drift/drift_analysis.py`
- output JSON: `logs/drift_analysis.json`
- visualization: `visualizations/drift_overview.png`

I used one small reference window and one production-style window. The production window was intentionally written to be longer, more complex, and slightly messier so the analysis would surface meaningful changes.

## Which Features Drifted Most

The strongest drift appeared in:

1. `token_count`
2. `char_count`
3. `avg_token_length`

The most important results were:

- token count mean
  - reference: `7.625`
  - production: `14.17`
  - drift score: `12.49`
  - p-value: `1.59e-05`

- character count mean
  - reference: `50.13`
  - production: `83.92`
  - drift score: `12.49`
  - p-value: `1.59e-05`

- average token length
  - reference: `5.66`
  - production: `4.92`
  - drift score: `1.75`
  - p-value: `0.214`

The first two are the clearest shift signals. The production queries are materially longer and more detailed than the reference queries. That matters because the system was originally evaluated on shorter, cleaner prompts.

The punctuation-based and multi-question features also increased, but they did not produce the same degree of formal drift in this small run. Even so, they remain operationally relevant because they are closely tied to input complexity and ambiguity.

## Integrity and Anomaly Findings

The anomaly summary from the production-style window was:

- multi-question requests: `2`
- high-punctuation requests: `3`
- very long queries: `1`
- low-diversity queries: `0`

These anomalies are not extreme in count, but they point to exactly the kind of request patterns that can make RAG systems harder to ground cleanly. Multi-question prompts and punctuation-heavy prompts often bundle several intents together, which can make the retriever return only partially relevant evidence.

## Expected Impact on Model Performance

The most likely downstream effect of this drift is not a total system failure but a gradual drop in grounding quality.

Longer and more complex queries can hurt the system in at least three ways:

1. retrieval may return a mix of partially relevant chunks rather than one clean cluster of evidence
2. the model may answer only part of the question while still sounding confident
3. the agent may have to compress more context into the same bounded tool workflow

This is important because the monitoring component already showed that the service is much more sensitive to generation behavior than to retrieval speed alone. If queries keep getting longer and more compound, the likely outcome is:

- more partial answers
- more missing citations
- higher generation latency
- lower task success on the agent side

## Recommended Intervention

My recommended action plan would be:

1. add a query-complexity watch threshold
   - alert when average token count or character count rises far above the evaluation baseline
2. route multi-question prompts differently
   - either split them upstream or force a more explicit clarification/retrieval policy
3. expand the drift monitoring window
   - the current script is a small proof of concept; in a real deployment this should run continuously
4. re-evaluate retrieval and prompt strategy under the new query mix
   - especially if long-form prompts become the norm

If this drift pattern persisted over time, I would not jump immediately to retraining a model. The first intervention would be to adjust retrieval and prompting behavior, because the drift is happening at the request-distribution level rather than in the underlying model weights.

## Conclusion

The main message from this drift analysis is that query complexity is drifting upward. That is the most operationally important finding because it directly affects the parts of the system that matter most in this architecture: retrieval quality, groundedness, and generation latency.

So the recommended response is:

- monitor query complexity continuously
- treat long and multi-question prompts as a meaningful operational risk
- update retrieval and prompt-handling rules before considering heavier model changes
