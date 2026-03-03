#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-file", default="outputs/metrics_latest.json")
    p.add_argument("--min-accuracy", type=float, default=0.90)
    p.add_argument("--min-f1", type=float, default=0.90)
    a = p.parse_args()

    mf = Path(a.metrics_file)
    if not mf.exists():
        print(f"FAILED: metrics file not found: {mf}")
        sys.exit(1)

    m = json.loads(mf.read_text())
    acc = float(m.get("accuracy", 0.0))
    f1 = float(m.get("f1_score", 0.0))

    failed = []
    if acc < a.min_accuracy:
        failed.append(f"accuracy {acc:.4f} < {a.min_accuracy:.4f}")
    if f1 < a.min_f1:
        failed.append(f"f1_score {f1:.4f} < {a.min_f1:.4f}")

    report = {"passed": len(failed)==0, "accuracy": acc, "f1_score": f1, "failures": failed}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    sys.exit(0 if report["passed"] else 1)

if __name__ == "__main__":
    main()
