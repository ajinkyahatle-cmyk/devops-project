import json

from insurance.metrics import evaluate, main, read_metrics


def test_gate_passes_and_fails():
    assert evaluate({"holdout_r2": 0.85}, 0.8) == []
    assert evaluate({"holdout_r2": 0.5}, 0.8)
    assert evaluate({}, 0.8)


def test_read_metrics_missing_returns_empty(tmp_path):
    assert read_metrics(tmp_path / "insurance_model") == {}


def test_cli_exit_codes(tmp_path):
    model = tmp_path / "insurance_model"
    (tmp_path / "metrics.json").write_text(json.dumps({"holdout_r2": 0.9}))
    assert main(["--model", str(model), "--min-r2", "0.8"]) == 0
    assert main(["--model", str(model), "--min-r2", "0.95"]) == 1
