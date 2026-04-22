# Agent Report

## Overview

For Part 2, I built a small multi-tool agent in `agent_controller.py`. The agent reuses the retriever from Part 1 and adds two additional tools:

- `summarize_context`
- `extract_evidence`

The model is responsible for deciding which action to take next, but the controller also constrains the decision space so the run stays interpretable and does not wander into repeated tool calls.

## Tool Selection Policy

The final policy is intentionally simple:

1. retrieve evidence first
2. if the task clearly asks for a summary, use `summarize_context`
3. if the task clearly asks for a list, bullets, or extracted facts, use `extract_evidence`
4. finish with a grounded answer

This version worked better than the earlier planner-only version. The first attempt allowed the model to keep choosing synthesis tools repeatedly, which produced traces that were technically observable but not efficient. Constraining the allowed action set by stage made the agent much more reliable without hiding the reasoning path.

## Retrieval Integration

The retriever is shared with `rag_pipeline.py`, so the agent is using the same embedding model, the same FAISS index, and the same document corpus as Part 1. After retrieval, the evidence is saved in a shared context buffer. That buffer is then reused by the summarization tool, the extraction tool, and the final answer step.

## Evaluation Tasks

The final measured run used 10 tasks, and the traces are saved in `agent_traces/`.

| Task ID | Outcome | Tools Used | Notes |
|---|---|---|---|
| task_01 | Success | retrieve → summarize_context → finish | RAG hallucination explanation |
| task_02 | Success | retrieve → extract_evidence → finish | Chunk-overlap design bullets |
| task_03 | Success | retrieve → finish | FAISS suitability explanation |
| task_04 | Success | retrieve → summarize_context → finish | Latency reporting summary |
| task_05 | Success | retrieve → extract_evidence → finish | Retrieval vs grounding distinction |
| task_06 | Success | retrieve → finish | Retrieval vs summarization policy |
| task_07 | Success | retrieve → summarize_context → finish | Highest-risk RAG failure modes |
| task_08 | Success | retrieve → extract_evidence → finish | Embedding tradeoff extraction |
| task_09 | Success | retrieve → extract_evidence → finish | Checklist-style trace answer |
| task_10 | Success | retrieve → summarize_context → finish | Chunking/indexing justification |

Aggregate results:

- Tasks evaluated: `10`
- Success rate: `1.00`
- Average duration: `6698.6 ms`
- Average step count: `1.9`

## Performance Analysis

The strongest pattern was a short workflow with one retrieval step followed by either direct answering or one additional synthesis step. That was enough to complete all 10 tasks successfully in the final run.

Tasks that naturally asked for a short explanation often worked well with `retrieve → finish`. Tasks that asked for bullets, key points, or extracted distinctions usually benefited from `extract_evidence`. Tasks that explicitly asked for a concise summary benefited from `summarize_context`.

The main failure mode in development was not wrong retrieval, but excessive looping. In the earlier version of the controller, the model sometimes kept choosing summarization or extraction after it already had enough context to answer. The final stage-aware policy fixed that problem.

## Model Analysis

The agent was evaluated with `meta-llama/llama-3.1-8b-instruct` through OpenRouter, using instructor approval for API-based inference. In practice, the model followed the JSON planning prompt reasonably well, but it performed much better once the controller limited the available actions at each stage.

This suggests that the model was capable enough for the assignment, but the quality of the agent depended heavily on the surrounding control logic. In other words, the model alone was not enough; the wrapper policy mattered.

## Latency Discussion

The average task duration in the final run was about `6.7 seconds`. The fastest task finished in about `3.9 seconds`, while the slowest took about `13.6 seconds`.

Most of that time came from the model calls rather than retrieval. Since retrieval is relatively cheap in this setup, the main tradeoff is between trace quality and the number of LLM turns per task. Keeping the workflow short improved both reliability and speed.

## Limitations

The current agent is designed for the course milestone, not as a general-purpose autonomous agent. It works best when tasks can be handled with one retrieval step and at most one synthesis step.

The corpus is still small and synthetic, so the agent benefits from a fairly controlled problem setting. A larger and messier document collection would likely require better routing, stronger retrieval quality, and a more robust stopping policy.
