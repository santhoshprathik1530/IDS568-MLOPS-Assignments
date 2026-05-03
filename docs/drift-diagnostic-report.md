# Drift Diagnostic Report

## Overview

For this project, I treated drift primarily as a change in the request distribution rather than as a label-shift problem on a supervised dataset. That framing fits the architecture better because the most important operational question is whether the incoming query mix is changing in ways that will hurt retrieval quality, groundedness, or agent behavior.

The analysis artifacts are:

- script: `src/drift/drift_analysis.py`
- structured output: `logs/drift_analysis.json`
- visualization: `visualizations/drift_overview.png`

I compared a small reference window with a production-style window that was intentionally longer, messier, and more complex.

## Which Features Drifted Most

The strongest drift appeared in:

1. `token_count`
2. `char_count`
3. `avg_token_length`

The clearest numerical results were:

- token count
  - reference mean: `7.625`
  - production mean: `14.17`
  - drift score: `12.49`
  - p-value: `1.59e-05`

- character count
  - reference mean: `50.13`
  - production mean: `83.92`
  - drift score: `12.49`
  - p-value: `1.59e-05`

- average token length
  - reference mean: `5.66`
  - production mean: `4.92`
  - drift score: `1.75`
  - p-value: `0.214`

The first two signals matter the most. They show that the production-style prompts are substantially longer and more detailed than the reference prompts. That is important because the system was initially evaluated on shorter, cleaner questions.

Punctuation-heavy prompts and multi-question prompts also increased, although in this small sample they did not produce the same level of formal statistical drift. Even so, they are operationally relevant because they often correspond to more ambiguous or more demanding retrieval conditions.

## Integrity and Anomaly Findings

The anomaly summary for the production-style window was:

- multi-question requests: `2`
- high-punctuation requests: `3`
- very long queries: `1`
- low-diversity queries: `0`

These counts are not extreme, but they point to the kinds of prompt patterns that can degrade a RAG system gradually rather than catastrophically. When users submit longer prompts that bundle multiple questions together, the retriever is more likely to return evidence that is only partially relevant.

## Impact on Model Performance

The likely effect of this drift is a drop in grounding quality rather than a complete system failure.

Longer and more complex prompts can harm the workflow in several ways:

1. retrieval may return a broader but less precise mix of chunks
2. the model may answer only part of a compound question while sounding complete
3. the agent may need to compress more context into a fixed and narrow tool sequence

This matters because the monitoring component already showed that generation dominates latency and that citation gaps are a meaningful signal. If the query mix continues drifting toward longer and more complex prompts, the expected outcome is:

- more partial answers
- more missing citations
- higher generation latency
- lower reliability for agent tasks

## Recommended Intervention

My recommended response would be operational rather than model-centric at first.

1. add a query-complexity watch threshold
   - monitor average token count and character count against the evaluation baseline
2. treat multi-question prompts as a separate handling case
   - split them when possible or enforce a stricter retrieval policy
3. extend the drift analysis from a batch script to a recurring check
   - the current implementation is a proof of concept, not a full monitoring service
4. re-evaluate retrieval and prompting under the newer query mix
   - especially if long-form prompts become normal usage

If this pattern persisted, I would not immediately retrain a model. The first response should be to adjust retrieval and prompt handling because the drift is occurring at the request-distribution level.

## Conclusion

The main conclusion is that query complexity is increasing. For this system, that is the most operationally significant form of drift because it affects retrieval precision, groundedness, and latency at the same time.

The practical response is therefore to monitor query complexity continuously, treat compound prompts as a real risk factor, and adapt retrieval and prompt handling before making heavier architectural changes.
