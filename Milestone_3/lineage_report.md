# Milestone 3 Lineage Report

## Scope
This report documents experiment lineage and model promotion decisions for `milestone3-iris-model` using MLflow tracking in `Milestone_3/mlflow.db`.

## Run Comparison (Part 3 Hyperparameter Sweep)
Experiment: `milestone3-train-fresh`  
Comparison group tag: `part3_fresh_hyperparam_sweep`

| Run Name | Run ID | n_estimators | max_depth | data_version | accuracy | f1_score | candidate_stage | model_sha256 (prefix) | model_md5 (prefix) |
|---|---|---:|---:|---|---:|---:|---|---|---|
| part3-fresh-hp1-ne80-md4 | 4201c09284f641d1ba441f9b98267e2a | 80 | 4 | part3_fresh_v1 | 0.9667 | 0.9666 | production_candidate | 7dc355973821d032 | 573f5d93547afcdc |
| part3-fresh-hp2-ne100-md5 | 395da3e8290845fda510464a2fc77751 | 100 | 5 | part3_fresh_v2 | 0.9333 | 0.9333 | staging_candidate | aa3fd79c337d70b3 | 6147670477deaeba |
| part3-fresh-hp3-ne120-md6 | 6cfdfd23e1b44783bea36b4b0776203c | 120 | 6 | part3_fresh_v3 | 0.9000 | 0.8997 | baseline | 04c8352b40798cfa | 717a7bd6bd8aeefd |
| part3-fresh-hp4-ne140-md7 | 729a39abcf09422da52440506df70764 | 140 | 7 | part3_fresh_v4 | 0.9000 | 0.8997 | challenger | 6c7bb43a2564ba73 | 3635f8e1cf97e3fd |
| part3-fresh-hp5-ne160-md8 | bcafabe1106842dcaea336bcbc01040c | 160 | 8 | part3_fresh_v5 | 0.9000 | 0.8997 | challenger | 5301aa11ed5a1abf | 4cca176ac94d7e22 |

## Selection Decision
Production candidate selected by highest accuracy, then weighted F1 as tiebreaker:
1. `part3-fresh-hp1-ne80-md4` (accuracy `0.9667`, F1 `0.9666`)
2. `part3-fresh-hp2-ne100-md5` (accuracy `0.9333`, F1 `0.9333`)
3. Remaining runs tied at `0.9000` accuracy and `0.8997` F1

Justification:
- Best predictive performance is from `hp1`.
- Artifact hashes and data version tags are logged for all five runs.
- Hyperparameter sweep is systematic (`n_estimators` and `max_depth` increase together).

## Registry Progression
Registered model: `milestone3-iris-model`

| Version | Stage | run_id | Tag `promotion_step` | Description |
|---|---|---|---|---|
| 11 | None | 10bcb3a9fcc94c11a4629721498c29ca | none | Checkpoint retained in `None` stage |
| 12 | Staging | 975ba398bd1a421ea9f3ec4c1ca2ba2f | staging | Validation candidate in Staging |
| 13 | Production | 8d22cd73d9f2436c82acd264abbc670e | production | Production candidate selected and promoted |

This satisfies the required registry progression: `None -> Staging -> Production`.

## Risks and Monitoring Needs
1. Data drift risk: Iris distribution is static in this exercise; real deployments should track feature and label drift.
2. Performance regression risk: Use quality gate thresholds (`accuracy >= 0.90`, `f1_score >= 0.90`) on recurring runs.
3. Operational risk: Keep model promotion behind explicit review gates.
4. Reproducibility risk: Keep tracking URI, dependency versions, and run metadata conventions pinned.
