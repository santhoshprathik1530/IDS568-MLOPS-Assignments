#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import prepare_dataset


def on_failure_callback(context):
    ti = context.get("task_instance")
    print(
        f"[FAILURE] task={ti.task_id if ti else 'unknown'} "
        f"try={ti.try_number if ti else 'n/a'} run_id={context.get('run_id')}"
    )


def _get_conf(context):
    dag_run = context.get("dag_run")
    return dag_run.conf if dag_run and dag_run.conf else {}


def preprocess_data(**context):
    conf = _get_conf(context)
    data_version = conf.get("data_version", f"airflow_{context['ds_nodash']}")
    return prepare_dataset(data_version=data_version)


def train_model(**context):
    conf = _get_conf(context)

    # Optional retry test hook: fail first attempt intentionally.
    # Set M3_FAIL_ONCE_TRAIN=true (default false below) to test retries.
    fail_once_enabled = os.getenv("M3_FAIL_ONCE_TRAIN", "false").lower() == "true"
    ti = context["ti"]
    if fail_once_enabled and ti.try_number == 1:
        raise RuntimeError("Intentional first-attempt failure in train_model to test retry.")

    data_version = conf.get("data_version", f"airflow_{context['ds_nodash']}")
    n_estimators = int(conf.get("n_estimators", 100))
    max_depth = int(conf.get("max_depth", 5))
    run_name = conf.get("run_name", f"airflow-rf-ne{n_estimators}-md{max_depth}-dv-{data_version}")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////opt/airflow/mlflow.db")
    exp = os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3-airflow")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "milestone3-iris-model")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--data-version",
            data_version,
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            exp,
            "--model-name",
            model_name,
            "--n-estimators",
            str(n_estimators),
            "--max-depth",
            str(max_depth),
            "--run-name",
            run_name,
        ],
        check=True,
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    e = client.get_experiment_by_name(exp)
    run = client.search_runs(
        [e.experiment_id], order_by=["attributes.start_time DESC"], max_results=1
    )[0]
    return {
        "run_id": run.info.run_id,
        "model_name": model_name,
        "tracking_uri": tracking_uri,
        "data_version": data_version,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


def register_model(**context):
    out = context["ti"].xcom_pull(task_ids="train_model")
    run_id, model_name, tracking_uri = out["run_id"], out["model_name"], out["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.run_id == run_id:
            print(f"Already registered: version {mv.version}")
            return {"model_version": str(mv.version), "already_registered": True}

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    reg = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
    client.update_model_version(
        name=model_name,
        version=str(reg.version),
        description=f"Registered by Airflow from run {run_id} (data_version={out.get('data_version')})",
    )
    client.set_model_version_tag(
        name=model_name,
        version=str(reg.version),
        key="source",
        value="airflow_dag",
    )
    return {"model_version": str(reg.version), "already_registered": False}


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,  # max_retries
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["milestone3", "airflow", "mlflow"],
) as dag:
    t1 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model)
    t3 = PythonOperator(task_id="register_model", python_callable=register_model)

    t1 >> t2 >> t3
