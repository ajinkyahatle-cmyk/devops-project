"""Model loading and prediction (the only module besides train that imports PyCaret)."""
from pycaret.regression import load_model, predict_model


def load(path="models/insurance_model"):
    return load_model(str(path), verbose=False)


def predict_charge(model, df) -> float:
    """Return the predicted charge for the first row of df.

    PyCaret 2.x named the output column 'Label'; 3.x names it 'prediction_label'.
    """
    out = predict_model(model, data=df)
    col = "prediction_label" if "prediction_label" in out.columns else "Label"
    return float(out[col].iloc[0])
