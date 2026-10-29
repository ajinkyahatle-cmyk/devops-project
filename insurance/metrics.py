"""Read model metrics and enforce the quality gate.

No PyCaret import, so the app sidebar, the tests and CI can use it cheaply.

Usage:  python -m insurance.metrics --min-r2 0.80   (exit code 1 if the gate fails)
"""
import argparse
import json
import os
import sys
from pathlib import Path


def read_metrics(model_path="models/insurance_model") -> dict:
    """Return metrics.json next to the model, or {} if it is missing/unreadable."""
    path = Path(model_path).parent / "metrics.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def build_id() -> str:
    """Commit this build came from: $GIT_SHA (Docker), else static/version.json, else 'dev'."""
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return json.loads(Path("static/version.json").read_text()).get("git_sha", "dev")
    except (OSError, ValueError):
        return "dev"


def evaluate(metrics: dict, min_r2: float) -> list[str]:
    """Return a list of gate failures (empty list means the model may ship)."""
    r2 = metrics.get("holdout_r2")
    if r2 is None:
        return ["holdout_r2 is missing from metrics"]
    if r2 < min_r2:
        return [f"holdout_r2 {r2:.3f} is below the required {min_r2:.3f}"]
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/insurance_model")
    ap.add_argument("--min-r2", type=float, default=0.80)
    args = ap.parse_args(argv)

    metrics = read_metrics(args.model)
    failures = evaluate(metrics, args.min_r2)
    print(json.dumps(metrics, indent=2))
    for failure in failures:
        print(f"QUALITY GATE FAILED: {failure}", file=sys.stderr)
    if not failures:
        print(f"Quality gate passed (min R2 {args.min_r2}).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
