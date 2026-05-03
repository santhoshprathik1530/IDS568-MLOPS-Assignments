#!/usr/bin/env python3
"""Multi-tool agent used for the Milestone 6 workflow.

The agent reuses the retriever from `rag_pipeline.py` and keeps the tool loop
simple: retrieve first, optionally run one synthesis tool, then finish with a
grounded answer.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.rag_pipeline import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, RAGPipeline


@dataclass
class ToolResult:
    """Structured result for any tool invocation."""

    tool_name: str
    success: bool
    duration_ms: float
    output: Any
    error: Optional[str] = None


@dataclass
class AgentStep:
    """One step in the agent trace."""

    step_number: int
    thought: str
    action: str
    action_input: str
    observation: Any
    duration_ms: float


@dataclass
class AgentTrace:
    """Observable multi-step execution trace."""

    task_id: str
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    status: str = "incomplete"

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "status": self.status,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "step_count": len(self.steps),
            "final_answer": self.final_answer,
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass
class EvaluationTask:
    """One multi-step agent evaluation task."""

    task_id: str
    task: str
    expected_tool_sequence: list[str]


class AgentTools:
    """Concrete tool implementations used by the agent."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.context_buffer = ""
        self.retrieved_sources: list[str] = []

    def _execute(self, tool_name: str, function: Callable[[str], Any], input_text: str) -> ToolResult:
        start = time.perf_counter()
        try:
            output = function(input_text)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                output=output,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                output=None,
                error=str(exc),
            )

    def retrieve(self, query: str) -> ToolResult:
        def _run(input_text: str) -> dict[str, Any]:
            chunks, latency_ms = self.pipeline.retrieve(input_text, top_k=3)
            self.context_buffer = "\n\n".join(
                f"[source: {item.chunk.source}] {item.chunk.text}" for item in chunks
            )
            self.retrieved_sources = [item.chunk.source for item in chunks]
            return {
                "retrieval_latency_ms": round(latency_ms, 2),
                "sources": self.retrieved_sources,
                "context": self.context_buffer,
            }

        return self._execute("retrieve", _run, query)

    def summarize_context(self, instruction: str) -> ToolResult:
        def _run(input_text: str) -> dict[str, Any]:
            if not self.context_buffer:
                raise RuntimeError("No context available. Retrieval should run before summarization.")
            bullets = [line.strip() for line in self.context_buffer.splitlines() if line.strip()]
            short = bullets[:4]
            summary = " ".join(item for item in short)
            return {
                "instruction": input_text,
                "summary": summary[:700],
                "source_count": len(set(self.retrieved_sources)),
            }

        return self._execute("summarize_context", _run, instruction)

    def extract_evidence(self, instruction: str) -> ToolResult:
        def _run(input_text: str) -> dict[str, Any]:
            if not self.context_buffer:
                raise RuntimeError("No context available. Retrieval should run before extraction.")
            lines = [line.strip() for line in self.context_buffer.splitlines() if line.strip()]
            facts = []
            for line in lines:
                if line.startswith("[source:"):
                    continue
                if len(facts) == 5:
                    break
                facts.append(line[:200])
            return {
                "instruction": input_text,
                "facts": facts,
                "sources": list(dict.fromkeys(self.retrieved_sources)),
            }

        return self._execute("extract_evidence", _run, instruction)


def parse_json_block(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""

    text = text.strip()
    if not text:
        raise ValueError("The model returned an empty planner response.")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse planner JSON from response: {text}")
    return json.loads(match.group(0))


def build_agent_tasks() -> list[EvaluationTask]:
    """Ten multi-step tasks aligned with the milestone rubric."""

    return [
        EvaluationTask("task_01", "Explain how RAG reduces hallucination, then summarize the answer in two sentences.", ["retrieve", "summarize_context", "finish"]),
        EvaluationTask("task_02", "Find the guidance on chunk overlap and extract the key design decisions as bullets.", ["retrieve", "extract_evidence", "finish"]),
        EvaluationTask("task_03", "Retrieve evidence for why FAISS IndexFlatL2 fits a small local project and produce a concise final answer.", ["retrieve", "finish"]),
        EvaluationTask("task_04", "Gather the latency measurement guidance, then summarize what should be reported separately.", ["retrieve", "summarize_context", "finish"]),
        EvaluationTask("task_05", "Find how grounding failures differ from retrieval failures and list the distinction clearly.", ["retrieve", "extract_evidence", "finish"]),
        EvaluationTask("task_06", "Retrieve when an agent should use retrieval versus summarization, then explain the policy.", ["retrieve", "finish"]),
        EvaluationTask("task_07", "Find common failure modes in RAG systems and summarize the highest-risk ones.", ["retrieve", "summarize_context", "finish"]),
        EvaluationTask("task_08", "Collect evidence about embedding model tradeoffs and extract the important points.", ["retrieve", "extract_evidence", "finish"]),
        EvaluationTask("task_09", "Retrieve what observable traces should include and provide a final checklist-style answer.", ["retrieve", "finish"]),
        EvaluationTask("task_10", "Find how chunking and indexing design choices should be justified, then summarize the answer.", ["retrieve", "summarize_context", "finish"]),
    ]


class MultiToolAgent:
    """LLM-driven controller that selects tools step by step."""

    def __init__(self, pipeline: RAGPipeline, max_steps: int = 4):
        self.pipeline = pipeline
        self.tools = AgentTools(pipeline)
        self.max_steps = max_steps

    def _desired_intermediate_tool(self, task: str) -> str:
        """Infer the most useful post-retrieval tool from the task text."""

        task_lower = task.lower()
        extract_markers = ["extract", "bullets", "bullet", "list", "checklist", "important points"]
        summarize_markers = ["summarize", "summary", "two sentences", "concise", "brief"]

        if any(marker in task_lower for marker in extract_markers):
            return "extract_evidence"
        if any(marker in task_lower for marker in summarize_markers):
            return "summarize_context"
        return "finish"

    def _allowed_actions(self, task: str, trace: AgentTrace) -> list[str]:
        """Constrain planner choices to a short, stage-aware set."""

        if not trace.steps:
            return ["retrieve"]

        completed_actions = [step.action for step in trace.steps]
        desired_tool = self._desired_intermediate_tool(task)

        if "retrieve" not in completed_actions:
            return ["retrieve"]

        if len(trace.steps) >= 2:
            return ["finish"]

        if desired_tool != "finish" and desired_tool not in completed_actions:
            return [desired_tool, "finish"]

        return ["finish"]

    def _planner_prompt(self, task: str, trace: AgentTrace, allowed_actions: list[str]) -> tuple[str, str]:
        history_lines = []
        for step in trace.steps:
            history_lines.append(
                f"Step {step.step_number}: action={step.action} input={step.action_input}\n"
                f"observation={json.dumps(step.observation, ensure_ascii=True)}"
            )
        history = "\n".join(history_lines) if history_lines else "No prior steps."
        system_prompt = (
            "You are a tool-using controller for an MLOps assignment. "
            "Choose exactly one next action as JSON. "
            "Allowed actions: retrieve, summarize_context, extract_evidence, finish. "
            "Use retrieval when external facts are needed. "
            "Use summarize_context or extract_evidence only after retrieval. "
            "When enough evidence is gathered, choose finish. "
            "Do not repeat a tool if the task can already be completed."
        )
        prompt = textwrap.dedent(
            f"""
            Task:
            {task}

            Available tools:
            - retrieve: search the vector store for supporting evidence
            - summarize_context: condense the evidence already retrieved
            - extract_evidence: pull a structured list of facts from current evidence
            - finish: provide the final answer using the accumulated evidence

            Trace so far:
            {history}

            Allowed actions for this step:
            {", ".join(allowed_actions)}

            Return strict JSON with keys:
            {{
              "thought": "brief reasoning",
              "action": "retrieve|summarize_context|extract_evidence|finish",
              "action_input": "tool input or final answer instruction"
            }}
            """
        ).strip()
        return system_prompt, prompt

    def _final_prompt(self, task: str) -> tuple[str, str]:
        system_prompt = (
            "You are a grounded assistant. Use the available evidence only. "
            "Cite sources inline using [source: filename]."
        )
        prompt = textwrap.dedent(
            f"""
            Task:
            {task}

            Evidence:
            {self.tools.context_buffer or "No evidence collected."}

            Write the final answer. If evidence is weak, say so explicitly.
            """
        ).strip()
        return system_prompt, prompt

    def _call_planner(self, task: str, trace: AgentTrace) -> dict[str, Any]:
        allowed_actions = self._allowed_actions(task, trace)
        system_prompt, prompt = self._planner_prompt(task, trace, allowed_actions)
        raw = self.pipeline.generator.generate(prompt=prompt, system_prompt=system_prompt)
        decision = parse_json_block(raw)
        action = str(decision.get("action", "")).strip()
        if action not in allowed_actions:
            decision["action"] = allowed_actions[0]
            if "action_input" not in decision or not str(decision["action_input"]).strip():
                decision["action_input"] = task
            if not str(decision.get("thought", "")).strip():
                decision["thought"] = f"Fallback to allowed action: {allowed_actions[0]}"
        return decision

    def _execute_tool(self, action: str, action_input: str) -> ToolResult:
        if action == "retrieve":
            return self.tools.retrieve(action_input)
        if action == "summarize_context":
            return self.tools.summarize_context(action_input)
        if action == "extract_evidence":
            return self.tools.extract_evidence(action_input)
        raise ValueError(f"Unsupported action: {action}")

    def run_task(self, evaluation_task: EvaluationTask) -> AgentTrace:
        trace = AgentTrace(task_id=evaluation_task.task_id, task=evaluation_task.task)
        start = time.perf_counter()

        for step_number in range(1, self.max_steps + 1):
            decision = self._call_planner(evaluation_task.task, trace)
            thought = str(decision.get("thought", "")).strip()
            action = str(decision.get("action", "")).strip()
            action_input = str(decision.get("action_input", evaluation_task.task)).strip() or evaluation_task.task

            if action == "finish":
                system_prompt, prompt = self._final_prompt(evaluation_task.task)
                final_answer = self.pipeline.generator.generate(prompt=prompt, system_prompt=system_prompt)
                trace.final_answer = final_answer
                trace.status = "success"
                trace.total_duration_ms = (time.perf_counter() - start) * 1000
                return trace

            tool_result = self._execute_tool(action, action_input)
            trace.add_step(
                AgentStep(
                    step_number=step_number,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation={
                        "success": tool_result.success,
                        "duration_ms": round(tool_result.duration_ms, 2),
                        "output": tool_result.output,
                        "error": tool_result.error,
                    },
                    duration_ms=tool_result.duration_ms,
                )
            )

        system_prompt, prompt = self._final_prompt(evaluation_task.task)
        trace.final_answer = self.pipeline.generator.generate(prompt=prompt, system_prompt=system_prompt)
        trace.status = "success_with_forced_finish"
        trace.total_duration_ms = (time.perf_counter() - start) * 1000
        return trace

    def evaluate(self, tasks: Optional[list[EvaluationTask]] = None, trace_dir: str = "agent_traces") -> list[AgentTrace]:
        tasks = tasks or build_agent_tasks()
        output_dir = Path(trace_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        traces: list[AgentTrace] = []
        for task in tasks:
            self.tools.context_buffer = ""
            self.tools.retrieved_sources = []
            trace = self.run_task(task)
            traces.append(trace)
            trace_path = output_dir / f"{task.task_id}.json"
            trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
        return traces


def summarize_traces(traces: list[AgentTrace]) -> str:
    """Create a compact text summary for evaluation."""

    if not traces:
        return "No traces generated."
    durations = [trace.total_duration_ms for trace in traces]
    success_rate = sum(1 for trace in traces if trace.status == "success") / len(traces)
    step_counts = [len(trace.steps) for trace in traces]
    return "\n".join(
        [
            f"tasks={len(traces)}",
            f"success_rate={success_rate:.2f}",
            f"avg_duration_ms={statistics.mean(durations):.1f}",
            f"avg_steps={statistics.mean(step_counts):.1f}",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Milestone 6 multi-tool agent.")
    parser.add_argument("--task", help="Run one custom task instead of the 10-task evaluation set.")
    parser.add_argument("--evaluate", action="store_true", help="Run all 10 evaluation tasks and export traces.")
    parser.add_argument("--trace-dir", default="agent_traces", help="Directory used for exported trace JSON files.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model name.")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="OpenRouter model name used by the agent planner and final answer step.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(embed_model_name=args.embed_model, llm_model_name=args.llm_model)
    pipeline.ingest()
    agent = MultiToolAgent(pipeline)

    if args.evaluate:
        traces = agent.evaluate(trace_dir=args.trace_dir)
        print(summarize_traces(traces))
        print(f"saved_traces={args.trace_dir}")
        return

    task = args.task or "Explain when a multi-tool agent should retrieve before summarizing."
    trace = agent.run_task(EvaluationTask("adhoc", task, []))
    print(json.dumps(trace.to_dict(), indent=2))


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
