# Agent Report

## Agent Policy

The agent in `agent_controller.py` uses an instruct model accessed through OpenRouter to choose between:

- `retrieve`
- `summarize_context`
- `extract_evidence`
- `finish`

Retrieval is used when the task requires external facts. Summarization and extraction are used only after evidence has been gathered.
The final controller uses a stage-aware policy to prevent looping: retrieve first, optionally call one synthesis tool, then force a grounded final answer. This preserved transparency while improving completion reliability.

## Retrieval Integration

The agent reuses the retriever from `rag_pipeline.py`, which keeps retrieval logic centralized and avoids duplication.
The retrieved evidence is stored in a shared context buffer and then reused by `summarize_context`, `extract_evidence`, and the final answer step.

## 10 Evaluation Tasks

Run:

```bash
python agent_controller.py --evaluate --trace-dir agent_traces
```

Measured trace directory used for the final report: `agent_traces/`

| Task ID | Outcome | Tools Used | Notes |
|---|---|---|---|
| task_01 | Success | retrieve → summarize_context → finish | Two-step synthesis for RAG hallucination explanation |
| task_02 | Success | retrieve → extract_evidence → finish | Extracted chunk-overlap design bullets |
| task_03 | Success | retrieve → finish | Direct factual answer after evidence collection |
| task_04 | Success | retrieve → summarize_context → finish | Summarized latency guidance cleanly |
| task_05 | Success | retrieve → extract_evidence → finish | Distinguished retrieval vs grounding failures |
| task_06 | Success | retrieve → finish | Retrieved agent policy and answered directly |
| task_07 | Success | retrieve → summarize_context → finish | Summarized highest-risk RAG failure modes |
| task_08 | Success | retrieve → extract_evidence → finish | Extracted embedding tradeoff points |
| task_09 | Success | retrieve → extract_evidence → finish | Produced checklist-style trace answer |
| task_10 | Success | retrieve → summarize_context → finish | Summarized justification for chunking/index choices |

Aggregate results:

- Tasks evaluated: `10`
- Success rate: `1.00`
- Average total duration: `6698.6 ms`
- Average step count: `1.9`

## Performance Analysis

- Success cases:
  - The strongest pattern was retrieval followed by exactly one synthesis action. That was enough to complete nearly all tasks without extra looping.
  - The agent performed especially well on tasks that clearly implied a post-retrieval operation such as summarization or extraction.
- Failure cases:
  - The earlier version of the controller overused the planner and sometimes repeated extraction or summarization without terminating. This was corrected with the stage-aware allowed-action policy now used in `agent_controller.py`.
  - Remaining quality risk is not task completion but answer depth, since the evidence corpus is small and domain-specific.
- Latency observations:
  - Average task duration was about `6.7s`.
  - The fastest task was `task_09` at about `3.9s`; the slowest was `task_04` at about `13.6s`.
  - Agent latency is driven mostly by LLM planner and final-answer calls, not by retrieval itself.

## Model Analysis

- Model used:
  - `meta-llama/llama-3.1-8b-instruct` through OpenRouter, using instructor approval for API-based inference.
- Quality tradeoffs:
  - The model followed structured JSON planning prompts well once the controller constrained allowed actions by stage.
  - It produced usable grounded final answers but still benefits from a tightly bounded tool policy to avoid extra deliberation.
- Latency tradeoffs:
  - API inference simplified setup but introduced more variable latency than a consistently provisioned local server might.
  - The agent remained practical because each task used only one retrieval step plus at most one synthesis step before finalization.

## Limitations

- Tool choice quality still depends on prompt discipline and controller guardrails.
- The agent uses a milestone-sized synthetic corpus rather than a larger production-style document store.
- The final report should still reference concrete examples from specific trace files during submission polish.
