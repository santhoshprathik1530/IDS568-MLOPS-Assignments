#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Dict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def prepare_dataset(output_dir="artifacts/preprocessed", data_version="iris_v1", test_size=0.2, random_state=42) -> Dict[str, str]:
    version_dir = Path(output_dir) / data_version
    version_dir.mkdir(parents=True, exist_ok=True)

    train_path = version_dir / "train.csv"
    test_path = version_dir / "test.csv"
    config_path = version_dir / "preprocessing_config.json"
    meta_path = version_dir / "dataset_meta.json"

    if train_path.exists() and test_path.exists() and config_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["idempotent_hit"] = True
        return meta

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["target"]
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    config = {
        "data_version": data_version,
        "dataset": "sklearn_iris",
        "target_column": "target",
        "feature_columns": [c for c in df.columns if c != "target"],
        "split": {"test_size": test_size, "random_state": random_state, "stratify": True},
    }
    config_path.write_text(json.dumps(config, indent=2))

    meta = {
        "data_version": data_version,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "config_path": str(config_path),
        "meta_path": str(meta_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_sha256": file_hash(train_path, "sha256"),
        "test_sha256": file_hash(test_path, "sha256"),
        "config_sha256": file_hash(config_path, "sha256"),
        "idempotent_hit": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="artifacts/preprocessed")
    p.add_argument("--data-version", default="iris_v1")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(prepare_dataset(a.output_dir, a.data_version, a.test_size, a.random_state), indent=2))
