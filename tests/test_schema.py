import pytest

from insurance.schema import FEATURES, build_input_df


def test_valid_input_builds_one_row():
    df = build_input_df(30, "male", 25.5, 2, True, "southwest")
    assert list(df.columns) == FEATURES
    assert len(df) == 1
    assert df.loc[0, "smoker"] == "yes"


def test_smoker_false_maps_to_no():
    assert build_input_df(30, "female", 25, 0, False, "northeast").loc[0, "smoker"] == "no"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(age=5),
        dict(sex="other"),
        dict(bmi=5),
        dict(children=-1),
        dict(region="mars"),
        dict(smoker="maybe"),
    ],
)
def test_invalid_input_raises(kwargs):
    base = dict(age=30, sex="male", bmi=25, children=0, smoker="no", region="southwest")
    base.update(kwargs)
    with pytest.raises(ValueError):
        build_input_df(**base)
