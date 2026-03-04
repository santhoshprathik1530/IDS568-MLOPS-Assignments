# Milestone 3: Workflow Automation and Experiment Tracking

## What this repository implements
1. Airflow DAG (`dags/train_pipeline.py`) with `preprocess_data -> train_model -> register_model`
2. CI/CD workflow (`.github/workflows/train_and_validate.yml`) that trains, validates, and fails on metric regression
3. MLflow tracking + model registry with logged params, metrics, artifacts, hashes, and staged model versions

## Setup
1. Create and activate a Python 3.11+ environment.
2. Install dependencies:
```bash
cd Milestone_3
pip install -r requirements.txt
```

## Run locally (single training run)
```bash
cd Milestone_3
python preprocess.py --data-version local_v1
python train.py \
  --experiment-name milestone3-train \
  --tracking-uri sqlite:///mlflow.db \
  --model-name milestone3-iris-model \
  --run-name local-rf-ne100-md5 \
  --data-version local_v1 \
  --n-estimators 100 \
  --max-depth 5
python model_validation.py --min-accuracy 0.90 --min-f1 0.90
```

## Run Airflow pipeline
Airflow DAG file: `Milestone_3/dags/train_pipeline.py`

Example trigger payload for hyperparameter experiments:
```json
{
  "data_version": "exp_v1",
  "n_estimators": 80,
  "max_depth": 4,
  "run_name": "airflow-rf-ne80-md4-exp_v1"
}
```

## Architecture and lineage guarantees
1. Preprocessing is idempotent by data version: if artifacts for a data version already exist, the task returns existing metadata.
2. Training logs complete lineage in MLflow:
   - Params: model hyperparameters and data version
   - Metrics: accuracy, F1, and loss
   - Artifacts: serialized model + preprocessing config/meta
   - Tags: git commit, dataset hashes, model hashes
3. Registration creates/updates model versions in MLflow registry.

## CI-based model governance
Workflow: `.github/workflows/train_and_validate.yml`
1. Installs pinned dependencies
2. Runs training with explicit run name
3. Executes `model_validation.py` as quality gate
4. Uploads outputs/artifacts even on failure for auditability

## Experiment tracking methodology
1. Use controlled hyperparameter sweeps (`n_estimators`, `max_depth`)
2. Assign explicit MLflow run names (`--run-name`)
3. Tag run group and candidate rank for lineage comparison
4. Compare runs by params, metrics, and artifact hashes before promotion

## Documentation map
1. Lineage analysis: `lineage_report.md`
2. Operational procedures: `operational_notes.md`
