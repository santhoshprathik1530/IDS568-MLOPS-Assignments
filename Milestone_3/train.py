#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pickle, subprocess
from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocess import prepare_dataset

def file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def train_model(n_estimators=100, max_depth=5, random_state=42, data_version="iris_v1",
                experiment_name="milestone3-train", tracking_uri="sqlite:///mlflow.db",
                model_name="milestone3-iris-model", output_dir="outputs"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    prep = prepare_dataset(data_version=data_version)
    train_df = pd.read_csv(prep["train_path"])
    test_df = pd.read_csv(prep["test_path"])

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="weighted"))

        mlflow.log_params({
            "n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state,
            "data_version": data_version, "train_rows": len(train_df), "test_rows": len(test_df),
            "model_name": model_name
        })
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        model_dir = Path("artifacts/models") / run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "model.pkl"
        with model_file.open("wb") as f:
            pickle.dump(model, f)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(model_file), artifact_path="model_files")
        mlflow.log_artifact(prep["config_path"], artifact_path="preprocess")
        mlflow.log_artifact(prep["meta_path"], artifact_path="preprocess")

        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_version": data_version,
            "model_sha256": file_hash(model_file, "sha256"),
            "model_md5": file_hash(model_file, "md5"),
            "train_sha256": prep["train_sha256"],
            "test_sha256": prep["test_sha256"],
            "config_sha256": prep["config_sha256"]
        })

        out = {"run_id": run_id, "accuracy": accuracy, "f1_score": f1}
        Path(output_dir, "metrics_latest.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--data-version", default="iris_v1")
    p.add_argument("--experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "milestone3-train"))
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    p.add_argument("--model-name", default=os.getenv("MLFLOW_MODEL_NAME", "milestone3-iris-model"))
    p.add_argument("--output-dir", default="outputs")
    a = p.parse_args()
    train_model(a.n_estimators, a.max_depth, a.random_state, a.data_version, a.experiment_name, a.tracking_uri, a.model_name, a.output_dir)
