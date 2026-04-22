#!/usr/bin/env python3
"""RAG pipeline used for Milestone 6.

The pipeline keeps the moving parts explicit:
chunk documents, embed them, index them in FAISS, retrieve top-k evidence,
and send a grounded prompt to a real instruct model.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
DEFAULT_TOP_K = 3


@dataclass
class Document:
    """A source document in the knowledge base."""

    doc_id: str
    title: str
    category: str
    source: str
    content: str


@dataclass
class Chunk:
    """A chunked segment with source metadata."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    category: str
    chunk_index: int
    text: str


@dataclass
class RetrievedChunk:
    """A retrieval result enriched with distance and score."""

    rank: int
    distance: float
    similarity_score: float
    chunk: Chunk


@dataclass
class AnswerResult:
    """Final RAG answer output."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    prompt: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float


@dataclass
class RetrievalEvaluationCase:
    """Ground-truth evaluation case for retrieval."""

    query_id: str
    question: str
    relevant_sources: list[str]
    expected_concepts: list[str]


@dataclass
class RetrievalEvaluationResult:
    """Per-query retrieval evaluation result."""

    query_id: str
    question: str
    retrieved_sources: list[str]
    relevant_sources: list[str]
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float
    answer_excerpt: str


@dataclass
class TimingSummary:
    """Aggregate timing metrics for repeated operations."""

    stage: str
    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return collapsed.strip()


def build_sample_corpus() -> list[Document]:
    """Return a compact but diverse MLOps/RAG knowledge base."""

    raw_documents = [
        {
            "doc_id": "doc_rag_architecture",
            "title": "RAG Architecture Overview",
            "category": "rag",
            "source": "rag_architecture.md",
            "content": """
            Retrieval-augmented generation combines retrieval with generative inference.
            Instead of answering only from parametric memory, the system first searches an
            external knowledge base, retrieves the most relevant chunks, and then asks the
            language model to answer using those chunks as grounding context.

            A typical pipeline includes ingestion, chunking, embedding generation, vector
            indexing, retrieval, prompt construction, and grounded answer generation. This
            architecture reduces hallucination because the model is constrained to reference
            retrieved evidence instead of relying only on pretraining.

            Production RAG systems benefit from source attribution, clear no-answer behavior,
            and instrumentation that separates retrieval latency from generation latency.
            """,
        },
        {
            "doc_id": "doc_chunking_design",
            "title": "Chunking Strategy Notes",
            "category": "retrieval",
            "source": "chunking_strategy.md",
            "content": """
            Chunking controls retrieval granularity. Smaller chunks improve precision because
            they contain less noise, but they may lose context. Larger chunks preserve more
            context, but can dilute relevance and consume prompt budget.

            A practical starting point for text-heavy documents is 350 to 700 words or a few
            hundred characters with overlap. Overlap helps preserve meaning at chunk boundaries.
            Recursive splitting usually prefers paragraph boundaries first, then sentence
            boundaries, and only falls back to fixed-size slicing when necessary.

            Design decisions should explain chunk size, overlap, and the rationale for those
            values based on document structure and context-window constraints.
            """,
        },
        {
            "doc_id": "doc_embeddings",
            "title": "Embedding Model Selection",
            "category": "retrieval",
            "source": "embedding_models.md",
            "content": """
            Embedding models convert text into dense vectors that capture semantic similarity.
            Sentence-transformer models such as all-MiniLM-L6-v2 are common starting points for
            coursework because they are lightweight, open-source, and fast enough for local use.

            Embeddings support semantic retrieval, meaning a query about hallucination control
            can still match chunks describing grounding even if the wording differs. Good
            embedding choices balance latency, vector dimension, and retrieval quality.
            """,
        },
        {
            "doc_id": "doc_vector_store",
            "title": "FAISS for Local Vector Search",
            "category": "vector-db",
            "source": "faiss_notes.md",
            "content": """
            FAISS is an open-source similarity search library for dense vectors. IndexFlatL2
            performs exact nearest-neighbor search using Euclidean distance and works well for
            small to moderate datasets because it is simple, deterministic, and easy to debug.

            FAISS expects float32 embeddings. For small milestone-scale corpora, exact search is
            usually acceptable and makes evaluation easier because retrieval results are stable.
            Approximate indexes become more attractive only when the corpus grows substantially.
            """,
        },
        {
            "doc_id": "doc_grounding",
            "title": "Grounding and Citation Practices",
            "category": "generation",
            "source": "grounding_practices.md",
            "content": """
            Grounded generation means the language model must answer only from retrieved context.
            Good prompts explicitly instruct the model to refuse unsupported claims, cite sources,
            and mention uncertainty when the evidence is incomplete.

            Grounding analysis should separate retrieval failures from generation failures. If the
            correct evidence is absent from retrieved chunks, the error belongs to retrieval. If
            the evidence is present but the model still invents facts or ignores citations, the
            error belongs to generation or instruction-following.
            """,
        },
        {
            "doc_id": "doc_latency",
            "title": "Latency Measurement Guidance",
            "category": "evaluation",
            "source": "latency_measurements.md",
            "content": """
            RAG latency should be measured at multiple stages. Retrieval latency covers query
            embedding plus vector search. Generation latency covers the model response time after
            prompt construction. End-to-end latency includes retrieval, prompt construction,
            generation, and any post-processing.

            Reporting only total runtime hides bottlenecks. Stage-level measurements help explain
            whether slow performance comes from embedding, vector lookup, or model inference.
            """,
        },
        {
            "doc_id": "doc_agents",
            "title": "Agent Tool-Use Policy",
            "category": "agent",
            "source": "agent_policy.md",
            "content": """
            A multi-tool agent should choose retrieval when a task depends on external facts.
            It should choose summarization when enough evidence has already been collected and the
            user needs a concise synthesis. It should choose extraction when the task requests a
            structured list of entities, metrics, or decisions from available evidence.

            Good agent traces log the decision, the chosen tool, the tool input, the output, and
            the reason the agent moves to the next step. Observable traces make failure analysis
            possible and satisfy milestone transparency requirements.
            """,
        },
        {
            "doc_id": "doc_failure_modes",
            "title": "Failure Modes in RAG and Agents",
            "category": "evaluation",
            "source": "failure_analysis.md",
            "content": """
            Common failure modes include poor chunk boundaries, ambiguous queries, retrieving
            tangential context, over-confident generation, and agents using the wrong tool too
            early. Another failure mode is stopping after retrieval without synthesizing a final
            answer that actually addresses the user task.

            Evaluation should document at least a few negative cases, including out-of-scope
            questions and ambiguous requests, because those reveal whether the system behaves
            safely when evidence is weak or missing.
            """,
        },
    ]
    return [Document(**{**item, "content": _normalize_whitespace(item["content"])}) for item in raw_documents]


def build_evaluation_queries() -> list[RetrievalEvaluationCase]:
    """Return 10 handcrafted evaluation questions with relevant sources."""

    return [
        RetrievalEvaluationCase(
            query_id="q1",
            question="What is retrieval-augmented generation and how does it reduce hallucination?",
            relevant_sources=["rag_architecture.md", "grounding_practices.md"],
            expected_concepts=["retrieval", "grounding", "hallucination"],
        ),
        RetrievalEvaluationCase(
            query_id="q2",
            question="Why does chunk overlap matter when building a retriever?",
            relevant_sources=["chunking_strategy.md"],
            expected_concepts=["overlap", "boundaries", "context"],
        ),
        RetrievalEvaluationCase(
            query_id="q3",
            question="Why is FAISS IndexFlatL2 a reasonable choice for a small course project corpus?",
            relevant_sources=["faiss_notes.md"],
            expected_concepts=["exact", "stable", "small dataset"],
        ),
        RetrievalEvaluationCase(
            query_id="q4",
            question="How should retrieval latency, generation latency, and end-to-end latency be measured?",
            relevant_sources=["latency_measurements.md"],
            expected_concepts=["retrieval latency", "generation latency", "end-to-end"],
        ),
        RetrievalEvaluationCase(
            query_id="q5",
            question="What makes an answer grounded instead of hallucinatory?",
            relevant_sources=["grounding_practices.md", "rag_architecture.md"],
            expected_concepts=["cite", "supported", "uncertainty"],
        ),
        RetrievalEvaluationCase(
            query_id="q6",
            question="How do sentence-transformer embeddings help semantic retrieval?",
            relevant_sources=["embedding_models.md"],
            expected_concepts=["semantic similarity", "vectors", "retrieval"],
        ),
        RetrievalEvaluationCase(
            query_id="q7",
            question="When should an agent retrieve versus summarize information?",
            relevant_sources=["agent_policy.md"],
            expected_concepts=["tool choice", "retrieve", "summarize"],
        ),
        RetrievalEvaluationCase(
            query_id="q8",
            question="What failure cases should be documented during RAG evaluation?",
            relevant_sources=["failure_analysis.md", "grounding_practices.md"],
            expected_concepts=["failure", "negative case", "retrieval vs generation"],
        ),
        RetrievalEvaluationCase(
            query_id="q9",
            question="What design decisions should be justified for chunking and indexing?",
            relevant_sources=["chunking_strategy.md", "faiss_notes.md"],
            expected_concepts=["chunk size", "overlap", "index choice"],
        ),
        RetrievalEvaluationCase(
            query_id="q10",
            question="What should agent traces record to make decision-making observable?",
            relevant_sources=["agent_policy.md"],
            expected_concepts=["decision", "tool input", "tool output", "reasoning"],
        ),
    ]


def sentence_split(text: str) -> list[str]:
    """Split text on common sentence boundaries while keeping plain Python only."""

    text = text.strip()
    if not text:
        return []
    normalized = text.replace("? ", "?\n").replace("! ", "!\n").replace(". ", ".\n")
    return [part.strip() for part in normalized.splitlines() if part.strip()]


def chunk_document(document: Document, max_chars: int = 550, overlap_chars: int = 120) -> list[Chunk]:
    """Chunk a document recursively by paragraphs, then sentences, then windows."""

    paragraphs = [part.strip() for part in document.content.split("\n\n") if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue

        sentences = sentence_split(paragraph)
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if len(sentence) <= max_chars:
                current = sentence
                continue

            start = 0
            while start < len(sentence):
                window = sentence[start : start + max_chars].strip()
                if window:
                    pieces.append(window)
                if start + max_chars >= len(sentence):
                    break
                start += max_chars - overlap_chars
            current = ""
        if current:
            pieces.append(current)

    merged: list[str] = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue
        previous = merged[-1]
        if len(previous) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{previous} {piece}".strip()
        else:
            merged.append(piece)

    chunks: list[Chunk] = []
    for idx, piece in enumerate(merged):
        prefix = ""
        if idx > 0 and overlap_chars > 0:
            prefix = merged[idx - 1][-overlap_chars:].strip()
        text = f"{prefix} {piece}".strip() if prefix else piece
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_{idx}",
                doc_id=document.doc_id,
                source=document.source,
                title=document.title,
                category=document.category,
                chunk_index=idx,
                text=text,
            )
        )
    return chunks


class LocalEmbeddingModel:
    """Lazy wrapper around sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)


class OpenRouterGenerator:
    """Generator that talks to OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required for OpenRouter-backed generation. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export your key before running the pipeline, "
                "for example `export OPENROUTER_API_KEY='...'`."
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
                    "X-Title": os.getenv("OPENROUTER_APP_NAME", "IDS568 Milestone 6"),
                },
                timeout=180,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Failed to contact OpenRouter. Check your OPENROUTER_API_KEY, model name, "
                "and network access."
            ) from exc

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {body}")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        return str(content).strip()


class RAGPipeline:
    """Complete RAG pipeline with retrieval, generation, and evaluation."""

    def __init__(
        self,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        top_k: int = DEFAULT_TOP_K,
        max_chars: int = 550,
        overlap_chars: int = 120,
    ):
        self.embedder = LocalEmbeddingModel(embed_model_name)
        self.generator = OpenRouterGenerator(llm_model_name)
        self.top_k = top_k
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self._timings: dict[str, list[float]] = {}

    def _record_timing(self, stage: str, duration_ms: float) -> None:
        self._timings.setdefault(stage, []).append(duration_ms)

    def _require_index(self):
        if self.index is None or self.embeddings is None or not self.chunks:
            raise RuntimeError("The pipeline has not been ingested yet. Call ingest() first.")

    def ingest(self, documents: Optional[list[Document]] = None) -> int:
        """Chunk documents, embed them, and build a FAISS index."""

        documents = documents or build_sample_corpus()
        self.documents = documents

        start = time.perf_counter()
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(chunk_document(document, self.max_chars, self.overlap_chars))
        self.chunks = all_chunks
        self._record_timing("chunking", (time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        self.embeddings = self.embedder.encode([chunk.text for chunk in self.chunks])
        self._record_timing("embedding", (time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required. Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        dimension = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype(np.float32))
        self._record_timing("indexing", (time.perf_counter() - start) * 1000)
        return len(self.chunks)

    def retrieve(self, question: str, top_k: Optional[int] = None) -> tuple[list[RetrievedChunk], float]:
        """Retrieve the most relevant chunks for a question."""

        self._require_index()
        top_k = top_k or self.top_k

        start = time.perf_counter()
        query_embedding = self.embedder.encode([question]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        duration_ms = (time.perf_counter() - start) * 1000
        self._record_timing("retrieval", duration_ms)

        results: list[RetrievedChunk] = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            chunk = self.chunks[int(idx)]
            similarity = 1.0 / (1.0 + float(distance))
            results.append(
                RetrievedChunk(
                    rank=rank,
                    distance=float(distance),
                    similarity_score=similarity,
                    chunk=chunk,
                )
            )
        return results, duration_ms

    def build_prompt(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, str]:
        """Create a grounded prompt with source attribution."""

        if not retrieved_chunks:
            context = "No supporting context was retrieved."
        else:
            blocks = []
            for item in retrieved_chunks:
                blocks.append(
                    "\n".join(
                        [
                            f"[Chunk {item.rank}]",
                            f"source: {item.chunk.source}",
                            f"title: {item.chunk.title}",
                            f"distance: {item.distance:.4f}",
                            f"content: {item.chunk.text}",
                        ]
                    )
                )
            context = "\n\n".join(blocks)

        system_prompt = (
            "You are a grounded assistant for an MLOps milestone. "
            "Answer only from the provided context. "
            "If the evidence is insufficient, say so explicitly. "
            "Cite sources inline using the format [source: filename]."
        )
        prompt = textwrap.dedent(
            f"""
            Context:
            {context}

            Question:
            {question}

            Instructions:
            1. Use only the context above.
            2. If the context is incomplete, state the limitation clearly.
            3. Support each major claim with one or more inline citations.
            4. Keep the answer concise but specific.
            """
        ).strip()
        return system_prompt, prompt

    def generate_answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, str, float]:
        """Run grounded generation using the local LLM."""

        system_prompt, prompt = self.build_prompt(question, retrieved_chunks)
        start = time.perf_counter()
        answer = self.generator.generate(prompt=prompt, system_prompt=system_prompt)
        duration_ms = (time.perf_counter() - start) * 1000
        self._record_timing("generation", duration_ms)
        return answer, prompt, duration_ms

    def answer(self, question: str, top_k: Optional[int] = None) -> AnswerResult:
        """Retrieve evidence and generate the final grounded answer."""

        start = time.perf_counter()
        retrieved_chunks, retrieval_ms = self.retrieve(question, top_k=top_k)
        answer, prompt, generation_ms = self.generate_answer(question, retrieved_chunks)
        total_ms = (time.perf_counter() - start) * 1000
        self._record_timing("end_to_end", total_ms)
        return AnswerResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            prompt=prompt,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            end_to_end_latency_ms=total_ms,
        )

    def timing_report(self) -> dict[str, TimingSummary]:
        """Return aggregate timing summaries."""

        report: dict[str, TimingSummary] = {}
        for stage, values in self._timings.items():
            report[stage] = TimingSummary(
                stage=stage,
                count=len(values),
                total_ms=sum(values),
                avg_ms=statistics.mean(values),
                min_ms=min(values),
                max_ms=max(values),
            )
        return report

    def evaluate(self, cases: Optional[list[RetrievalEvaluationCase]] = None, top_k: Optional[int] = None) -> list[RetrievalEvaluationResult]:
        """Evaluate retrieval and grounded generation across 10 handcrafted queries."""

        cases = cases or build_evaluation_queries()
        results: list[RetrievalEvaluationResult] = []
        for case in cases:
            answer = self.answer(case.question, top_k=top_k)
            retrieved_sources = [item.chunk.source for item in answer.retrieved_chunks]
            relevant = set(case.relevant_sources)
            retrieved_set = set(retrieved_sources)
            true_positive = len(relevant & retrieved_set)
            precision = true_positive / max(len(retrieved_sources), 1)
            recall = true_positive / max(len(relevant), 1)
            hit_rate = 1.0 if true_positive > 0 else 0.0
            results.append(
                RetrievalEvaluationResult(
                    query_id=case.query_id,
                    question=case.question,
                    retrieved_sources=retrieved_sources,
                    relevant_sources=case.relevant_sources,
                    precision_at_k=precision,
                    recall_at_k=recall,
                    hit_rate=hit_rate,
                    retrieval_latency_ms=answer.retrieval_latency_ms,
                    generation_latency_ms=answer.generation_latency_ms,
                    end_to_end_latency_ms=answer.end_to_end_latency_ms,
                    answer_excerpt=answer.answer[:240],
                )
            )
        return results


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    """Human-readable retrieval display."""

    lines = []
    for item in chunks:
        lines.append(
            f"{item.rank}. {item.chunk.source} | {item.chunk.title} | "
            f"distance={item.distance:.4f} | score={item.similarity_score:.4f}"
        )
    return "\n".join(lines)


def format_evaluation_table(results: list[RetrievalEvaluationResult]) -> str:
    """Plain-text evaluation table."""

    lines = [
        "query_id | precision@k | recall@k | hit_rate | retrieval_ms | generation_ms | end_to_end_ms",
        "-" * 94,
    ]
    for item in results:
        lines.append(
            f"{item.query_id:7} | {item.precision_at_k:11.2f} | {item.recall_at_k:8.2f} | "
            f"{item.hit_rate:8.2f} | {item.retrieval_latency_ms:12.1f} | "
            f"{item.generation_latency_ms:13.1f} | {item.end_to_end_latency_ms:13.1f}"
        )
    if results:
        lines.append("-" * 94)
        lines.append(
            f"average | {statistics.mean(x.precision_at_k for x in results):11.2f} | "
            f"{statistics.mean(x.recall_at_k for x in results):8.2f} | "
            f"{statistics.mean(x.hit_rate for x in results):8.2f} | "
            f"{statistics.mean(x.retrieval_latency_ms for x in results):12.1f} | "
            f"{statistics.mean(x.generation_latency_ms for x in results):13.1f} | "
            f"{statistics.mean(x.end_to_end_latency_ms for x in results):13.1f}"
        )
    return "\n".join(lines)


def export_evaluation_json(results: list[RetrievalEvaluationResult], path: Path) -> None:
    """Persist evaluation outputs for later reporting."""

    payload = [asdict(item) for item in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_timing_report(report: dict[str, TimingSummary]) -> None:
    """Pretty-print timing metrics."""

    print("\nTiming summary")
    print("-" * 60)
    for stage, metrics in report.items():
        print(
            f"{stage:12} count={metrics.count:<3d} avg={metrics.avg_ms:8.1f}ms "
            f"min={metrics.min_ms:8.1f}ms max={metrics.max_ms:8.1f}ms"
        )


def run_demo(args: argparse.Namespace) -> None:
    """Run interactive demo or evaluation."""

    pipeline = RAGPipeline(
        embed_model_name=args.embed_model,
        llm_model_name=args.llm_model,
        top_k=args.top_k,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    num_chunks = pipeline.ingest()
    print(f"Ingested {len(pipeline.documents)} documents into {num_chunks} chunks.")

    if args.evaluate:
        results = pipeline.evaluate()
        print()
        print(format_evaluation_table(results))
        if args.export_json:
            export_evaluation_json(results, Path(args.export_json))
            print(f"\nSaved evaluation results to {args.export_json}")
        print_timing_report(pipeline.timing_report())
        return

    result = pipeline.answer(args.question, top_k=args.top_k)
    print("\nRetrieved chunks")
    print("-" * 60)
    print(format_retrieved_chunks(result.retrieved_chunks))
    print("\nAnswer")
    print("-" * 60)
    print(result.answer)
    print_timing_report(pipeline.timing_report())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Milestone 6 RAG pipeline.")
    parser.add_argument(
        "--question",
        default="What is retrieval-augmented generation and why does it reduce hallucination?",
        help="Question to answer with RAG.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run the 10-query evaluation set.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Sentence-transformer embedding model.")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="OpenRouter model name for grounded generation.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument("--max-chars", type=int, default=550, help="Maximum chunk size in characters.")
    parser.add_argument("--overlap-chars", type=int, default=120, help="Overlap size in characters.")
    parser.add_argument("--export-json", help="Optional JSON file path for evaluation output.")
    return parser


if __name__ == "__main__":
    run_demo(build_arg_parser().parse_args())
