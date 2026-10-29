"""Input schema and validation.

Pure pandas (no PyCaret import) so it is fast to unit-test and the UI can
validate input before the heavy model is touched.
"""
import pandas as pd

SEXES = ("male", "female")
REGIONS = ("southwest", "northwest", "northeast", "southeast")
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]


def build_input_df(age, sex, bmi, children, smoker, region) -> pd.DataFrame:
    """Validate raw inputs and return a one-row DataFrame the model accepts."""
    if not 18 <= age <= 100:
        raise ValueError(f"age must be between 18 and 100, got {age}")
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    if not 10 <= bmi <= 60:
        raise ValueError(f"bmi must be between 10 and 60, got {bmi}")
    if not 0 <= children <= 10:
        raise ValueError(f"children must be between 0 and 10, got {children}")
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    if isinstance(smoker, str):
        if smoker not in ("yes", "no"):
            raise ValueError(f"smoker must be 'yes' or 'no', got {smoker!r}")
    else:
        smoker = "yes" if smoker else "no"

    row = {
        "age": int(age),
        "sex": sex,
        "bmi": float(bmi),
        "children": int(children),
        "smoker": smoker,
        "region": region,
    }
    return pd.DataFrame([row], columns=FEATURES)
