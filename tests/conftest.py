import pytest


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One fast real PyCaret training run shared by all model tests."""
    pytest.importorskip("pycaret")
    from insurance.predict import load
    from insurance.train import train

    path = tmp_path_factory.mktemp("model") / "insurance_model"
    metrics = train("data/insurance.csv", path, fast=True, track=False)
    return path, load(path), metrics
