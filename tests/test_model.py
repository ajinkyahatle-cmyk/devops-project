"""Checks the artifacts and behaviour of a real (fast) PyCaret training run."""
import json

import pytest

pytest.importorskip("pycaret")

from insurance.predict import predict_charge
from insurance.schema import build_input_df


def test_artifacts_written(trained):
    path, _, _ = trained
    assert path.with_suffix(".pkl").exists()
    assert json.loads((path.parent / "metrics.json").read_text())["best_model"]
    assert (path.parent / "leaderboard.csv").exists()


def test_quality_gate(trained):
    _, _, metrics = trained
    assert metrics["holdout_r2"] > 0.7


def test_smoker_costs_more(trained):
    _, model, _ = trained
    base = {"age": 40, "sex": "male", "bmi": 28, "children": 1, "region": "southwest"}
    smoker = predict_charge(model, build_input_df(smoker=True, **base))
    non_smoker = predict_charge(model, build_input_df(smoker=False, **base))
    assert smoker > non_smoker
