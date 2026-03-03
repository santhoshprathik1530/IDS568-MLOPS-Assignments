#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import prepare_dataset
import mlflow
from mlflow.tracking import MlflowClient
import subprocess

def on_failure_callback(context):
    ti = context.get("task_instance")
    print(f"[FAILURE] task={ti.task_id if ti else 'unknown'} run_id={context.get('run_id')}")

def preprocess_data(**context):
    data_version = f"airflow_{context['ds_nodash']}"
    meta = prepare_dataset(data_version=data_version)
    return meta

def train_model(**context):
    data_version = f"airflow_{context['ds_nodash']}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3-airflow")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "milestone3-iris-model")

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--data-version", data_version,
        "--tracking-uri", tracking_uri,
        "--experiment-name", experiment_name,
        "--model-name", model_name
    ]
    subprocess.run(cmd, check=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No MLflow run found after training")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "tracking_uri": tracking_uri,
        "accuracy": run.data.metrics.get("accuracy", None),
        "f1_score": run.data.metrics.get("f1_score", None),
    }

def register_model(**context):
    train_out = context["ti"].xcom_pull(task_ids="train_model")
    run_id = train_out["run_id"]
    model_name = train_out["model_name"]
    tracking_uri = train_out["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # idempotent: if this run is already registered, skip creating another version
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == run_id:
            print(f"Run {run_id} already registered as version {mv.version}")
            return {"model_name": model_name, "model_version": str(mv.version), "already_registered": True}

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    reg = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    version = str(reg.version)
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Airflow registered run {run_id}"
    )
    client.set_model_version_tag(name=model_name, version=version, key="source", value="airflow_dag")

    print(f"Registered model {model_name} version {version} from run {run_id}")
    return {"model_name": model_name, "model_version": version, "already_registered": False}

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Milestone 3 DAG: preprocess -> train -> register",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["milestone3"],
) as dag:
    preprocess_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    register_task = PythonOperator(task_id="register_model", python_callable=register_model)

    preprocess_task >> train_task >> register_task
