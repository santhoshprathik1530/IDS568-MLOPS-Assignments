# IDS 568 Final Project

## Overview

This repository contains my final project for IDS 568. I used my Milestone 6 submission as the starting point and expanded it into a small but fully instrumented AI system with monitoring, experimentation, governance documentation, drift analysis, and risk review.

The underlying system is a retrieval-augmented generation workflow with a lightweight agent layer. Retrieval is handled with sentence-transformer embeddings and a FAISS index. Generation is handled by `meta-llama/llama-3.1-8b-instruct` through OpenRouter. I kept the scope intentionally narrow so that the final project could focus on operational behavior rather than on scaling a large application.

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

- Specification: [docs/experiment-specification.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/experiment-specification.md)
- Simulation: [src/ab_test/simulation.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/ab_test/simulation.py)
- Results: [logs/ab_test_results.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/ab_test_results.json), [visualizations/ab_test_summary.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/visualizations/ab_test_summary.png)
- Recommendation: [docs/recommendation-memo.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/recommendation-memo.md)

### Component 3: Model Card and Governance Packet

- Model card: [docs/model-card.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/model-card.md)
- Lineage diagram: [docs/lineage-diagram.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/lineage-diagram.png)
- Risk register: [docs/risk-register.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/risk-register.md)
- Audit trail: [logs/audit-trail.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/audit-trail.json)

### Component 4: Data Integrity and Drift Detection

- Script: [src/drift/drift_analysis.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/drift/drift_analysis.py)
- Outputs: [logs/drift_analysis.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/logs/drift_analysis.json), [visualizations/drift_overview.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/visualizations/drift_overview.png)
- Report: [docs/drift-diagnostic-report.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/drift-diagnostic-report.md)

### Component 5: AI Risk Assessment and Reflective Summary

- Governance review: [docs/governance-review.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/governance-review.md)
- Risk matrix: [docs/risk-matrix.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/risk-matrix.md)
- System boundary diagram: [docs/system-boundary-diagram.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/system-boundary-diagram.png)
- CTO memo: [docs/cto-memo.md](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/docs/cto-memo.md)

## Setup

I ran the project with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects these environment variables:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

## Reproduction

To regenerate the main artifacts:

```bash
python3 src/monitoring/run_monitoring_demo.py
python3 src/ab_test/simulation.py
python3 src/drift/drift_analysis.py
```

These commands write outputs to `logs/`, `screenshots/`, and `visualizations/`.

To re-run the Python syntax check:

```bash
python3 -m py_compile $(find . -name '*.py')
```

## System Notes

The knowledge base is intentionally small and is embedded directly in [src/common/rag_pipeline.py](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/src/common/rag_pipeline.py). It contains eight short documents covering RAG architecture, chunking, embeddings, FAISS retrieval, grounding practice, latency measurement, agent policy, and failure analysis. I used this narrow corpus on purpose so that the monitoring and governance components would be easy to interpret.

For the monitoring stack, I created both a Prometheus configuration file and a Grafana dashboard export:

- [dashboards/prometheus.yml](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/prometheus.yml)
- [dashboards/grafana-dashboard.json](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/dashboards/grafana-dashboard.json)

The corresponding screenshot is stored in [screenshots/dashboard-monitoring.png](/Users/santhoshkasam/Downloads/UIC/IDS%20568%20-%20MlOps/IDS568-MLOPS-Assignments/screenshots/dashboard-monitoring.png).

## Reflection and Lessons Learned

The most useful lesson from this course sequence was that a model can appear to work reasonably well until it is placed inside a larger operational setting. Once I added monitoring, drift checks, governance documentation, and an A/B framework, the system’s real weaknesses became easier to see. In my case, the main issues were not retrieval speed or missing infrastructure. They were more subtle: incomplete grounding, generation latency, and changes in query complexity over time.

Milestone 6 was the right foundation for this final project because a RAG pipeline with an agent naturally raises the kinds of questions this assignment is trying to surface. The monitoring dashboard connects directly to the drift analysis, the A/B test supports a concrete retrieval change, and the governance documents reflect the same system boundaries that show up in the code. That made the final submission feel less like five separate assignments and more like one coherent review of a deployed AI workflow.
