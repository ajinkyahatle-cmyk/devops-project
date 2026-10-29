"""Train, compare, tune and package the insurance model.

Pipeline:  setup -> compare_models -> tune_model (top N) -> automl -> hold-out
           score -> plots -> finalize_model -> save_model

Usage:  python -m insurance.train [--fast] [--no-track]
Experiment tracking uses MLflow; point MLFLOW_TRACKING_URI at a server, or leave
it unset to log to ./mlruns (browse with `make mlflow`).
"""
import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pycaret.regression import (
    automl,
    compare_models,
    finalize_model,
    models,
    plot_model,
    predict_model,
    pull,
    save_model,
    setup,
    tune_model,
)

log = logging.getLogger("insurance.train")

EXPERIMENT_NAME = "insurance-premium"
CANDIDATES = ["lr", "ridge", "lasso", "en", "rf", "et", "gbr", "lightgbm"]
FAST_CANDIDATES = ["lr", "gbr"]  # used by unit tests and quick local runs


def estimator_name(model) -> str:
    """PyCaret 3 returns a sklearn Pipeline; the last step is the real estimator."""
    est = model[-1] if hasattr(model, "steps") else model
    return type(est).__name__


def _save_plots(model, out_dir: Path) -> list[str]:
    """Best-effort diagnostic plots. Failures are logged, never fatal."""
    saved = []
    for plot in ("residuals", "feature"):
        try:
            tmp = plot_model(model, plot=plot, save=True, verbose=False)
            target = out_dir / f"{plot}.png"
            shutil.move(str(tmp), target)
            saved.append(target.name)
        except Exception as exc:  # plotting is optional (headless backends differ)
            log.warning("could not save %s plot: %s", plot, exc)
    return saved


def train(
    data_path="data/insurance.csv",
    model_path="models/insurance_model",
    seed=123,
    fast=False,
    track=True,
) -> dict:
    data = pd.read_csv(data_path)
    out_dir = Path(model_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    git_sha = os.getenv("GITHUB_SHA", "local")

    # 1. Experiment: split, encode, scale, add polynomial features.
    #    (PyCaret 3 removed trigonometry_features and feature_interaction.)
    tracking = {}
    if track:
        tracking = {
            "log_experiment": True,  # MLflow
            "experiment_name": EXPERIMENT_NAME,
            "experiment_custom_tags": {"git_sha": git_sha},
        }
    setup(
        data,
        target="charges",
        session_id=seed,
        fold=5,
        normalize=True,
        polynomial_features=True,
        html=False,
        verbose=False,
        **tracking,
    )

    # 2. Compare candidate algorithms by cross-validated R2.
    wanted = FAST_CANDIDATES if fast else CANDIDATES
    candidates = [m for m in wanted if m in models().index]
    top = compare_models(
        include=candidates, sort="R2", n_select=min(3, len(candidates)), verbose=False
    )
    top = top if isinstance(top, list) else [top]
    leaderboard = pull()
    leaderboard.to_csv(out_dir / "leaderboard.csv")

    # 3. Tune the top performers; keep a tuned model only if it beats the original.
    for model in top:
        tune_model(
            model, optimize="R2", n_iter=3 if fast else 20, choose_better=True, verbose=False
        )

    # 4. Best model across everything trained in this session (by CV R2).
    best = automl(optimize="R2")

    # 5. Honest estimate: score the untouched hold-out split.
    predict_model(best)
    holdout = pull()
    plots = _save_plots(best, out_dir)

    # 6. Refit on 100% of the data and save preprocessing + model as ONE pickle.
    final = finalize_model(best)
    save_model(final, str(model_path))

    metrics = {
        "best_model": estimator_name(best),
        "holdout_r2": float(holdout["R2"].iloc[0]),
        "holdout_mae": float(holdout["MAE"].iloc[0]),
        "holdout_rmse": float(holdout["RMSE"].iloc[0]),
        "rows": len(data),
        "candidates": candidates,
        "plots": plots,
        "seed": seed,
        "git_sha": git_sha,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/insurance.csv")
    ap.add_argument("--model", default="models/insurance_model")
    ap.add_argument("--fast", action="store_true", help="2 candidates, 3 tuning iterations")
    ap.add_argument("--no-track", action="store_true", help="disable MLflow logging")
    args = ap.parse_args()
    result = train(args.data, args.model, fast=args.fast, track=not args.no_track)
    print(json.dumps(result, indent=2))
