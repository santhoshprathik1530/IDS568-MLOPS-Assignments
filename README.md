# IDS 568 Final Project

## Overview

This repository contains my Module 8 final project for IDS 568. I built the project on top of my Milestone 6 RAG system and turned it into a monitored, evaluated, and governed AI service. The base system combines:

- a retrieval-augmented generation pipeline
- a lightweight multi-tool agent
- an operational layer for monitoring, experimentation, drift analysis, and risk review

The main model used for generation is `meta-llama/llama-3.1-8b-instruct` accessed through OpenRouter. Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` embeddings with a FAISS vector index.

## Repository Layout

```text
src/
  common/       base RAG pipeline and agent controller
  monitoring/   Component 1 instrumentation and monitoring demo
  ab_test/      Component 2 experiment simulation
  drift/        Component 4 drift and anomaly detection
docs/           written deliverables
dashboards/     Prometheus and Grafana configuration
logs/           generated metrics, audit trail, and experiment outputs
visualizations/ charts for drift and A/B analysis
screenshots/    dashboard evidence
```

## Deliverables By Component

### Component 1: Production Monitoring Dashboard

- Code: [src/monitoring/service.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/monitoring/service.py), [src/monitoring/traffic_simulator.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/monitoring/traffic_simulator.py), [src/monitoring/run_monitoring_demo.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/monitoring/run_monitoring_demo.py)
- Config: [dashboards/prometheus.yml](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/prometheus.yml), [dashboards/grafana-dashboard.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/grafana-dashboard.json)
- Evidence: [screenshots/dashboard-monitoring.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/screenshots/dashboard-monitoring.png), [logs/monitoring_summary.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/monitoring_summary.json)
- Interpretation: [docs/dashboard-interpretation.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/dashboard-interpretation.md)

### Component 2: A/B Test Design and Simulation

- Spec: [docs/experiment-specification.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/experiment-specification.md)
- Simulation: [src/ab_test/simulation.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/ab_test/simulation.py)
- Results: [logs/ab_test_results.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/ab_test_results.json), [visualizations/ab_test_summary.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/visualizations/ab_test_summary.png)
- Recommendation: [docs/recommendation-memo.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/recommendation-memo.md)

### Component 3: Model Card and Governance Packet

- Model card: [docs/model-card.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/model-card.md)
- Lineage diagram: [docs/lineage-diagram.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/lineage-diagram.png)
- Risk register: [docs/risk-register.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/risk-register.md)
- Audit trail: [logs/audit-trail.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/audit-trail.json)

### Component 4: Data Integrity and Drift Detection

- Scripts: [src/drift/drift_analysis.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/drift/drift_analysis.py)
- Outputs: [logs/drift_analysis.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/drift_analysis.json), [visualizations/drift_overview.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/visualizations/drift_overview.png)
- Report: [docs/drift-diagnostic-report.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/drift-diagnostic-report.md)

### Component 5: AI Risk Assessment and Reflective Summary

- Governance review: [docs/governance-review.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/governance-review.md)
- Risk matrix: [docs/risk-matrix.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/risk-matrix.md)
- System boundary diagram: [docs/system-boundary-diagram.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/system-boundary-diagram.png)
- Executive memo: [docs/cto-memo.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/cto-memo.md)

## Setup

Use Python 3.12 or a nearby version.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the model configuration:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

## Reproduction

### Run the monitoring demo

```bash
python3 src/monitoring/run_monitoring_demo.py
```

This generates the monitoring artifacts in `logs/` and the dashboard evidence in `screenshots/`.

### Run the A/B simulation

```bash
python3 src/ab_test/simulation.py
```

This writes the experiment summary to `logs/ab_test_results.json` and the chart to `visualizations/ab_test_summary.png`.

### Run the drift analysis

```bash
python3 src/drift/drift_analysis.py
```

This writes the structured output to `logs/drift_analysis.json` and the chart to `visualizations/drift_overview.png`.

### Re-run syntax validation

```bash
python3 -m py_compile $(find . -name '*.py')
```

## System Notes

The knowledge base comes from the small course-focused corpus embedded in [src/common/rag_pipeline.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/common/rag_pipeline.py). It includes eight source documents covering RAG architecture, chunking, embeddings, FAISS retrieval, grounding practices, latency notes, agent policy, and failure analysis.

For monitoring, I created both:

- a Prometheus collector configuration in [dashboards/prometheus.yml](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/prometheus.yml)
- a Grafana dashboard definition in [dashboards/grafana-dashboard.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/grafana-dashboard.json)

So yes, the project does include a Grafana dashboard artifact, along with a rendered dashboard screenshot.

## Reflection and Lessons Learned

The final project pulled together the ideas from the earlier milestones in a more realistic way than treating them as separate assignments. Milestone 6 was the most useful starting point because a RAG pipeline plus an agent already exposes the kinds of operational risks that the final project is asking us to reason about: latency, weak grounding, drift in the input mix, and tool-use failure modes.

The main lesson across the milestones was that building the model or pipeline is only part of the job. Once the system is wrapped with monitoring, governance, and evaluation, the weak points become much clearer. In this project, raw retrieval speed was not the main issue. Generation latency, missing citations, and more complex user queries were more meaningful operational signals.

Another takeaway was that the five components work better when they tell one consistent story. The monitoring metrics point to the same risks discussed in the governance review. The A/B test evaluates a change that is relevant to the system described in the model card. The drift analysis focuses on the same request patterns that would show up in the dashboard. That made the final project feel more like a small production system review than a stack of disconnected documents.
