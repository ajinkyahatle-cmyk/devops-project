import os
from pathlib import Path

import streamlit as st

from insurance.metrics import build_id, read_metrics
from insurance.predict import load, predict_charge
from insurance.schema import REGIONS, SEXES, build_input_df

MODEL_PATH = os.getenv("MODEL_PATH", "models/insurance_model")
HOSPITAL_IMG = Path("assets/hospital.jpg")


@st.cache_resource
def get_model():
    return load(MODEL_PATH)


def sidebar():
    st.sidebar.info("Built with PyCaret and Streamlit")
    if HOSPITAL_IMG.exists():
        st.sidebar.image(str(HOSPITAL_IMG))
    meta = read_metrics(MODEL_PATH)
    if meta:
        st.sidebar.markdown(
            f"**Model:** {meta.get('best_model', '?')}  \n"
            f"**Hold-out R²:** {meta.get('holdout_r2', float('nan')):.3f}  \n"
            f"**Hold-out MAE:** ${meta.get('holdout_mae', float('nan')):,.0f}"
        )
    st.sidebar.caption(f"Build {build_id()[:7]}")


def main():
    st.set_page_config(page_title="Insurance Premium Predictor")
    sidebar()
    st.title("Insurance Premium Predictor")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", SEXES)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.selectbox("Children", list(range(11)))
    smoker = st.checkbox("Smoker")
    region = st.selectbox("Region", REGIONS)

    if st.button("Predict"):
        try:
            df = build_input_df(age, sex, bmi, children, smoker, region)
            charge = predict_charge(get_model(), df)
        except ValueError as e:
            st.error(str(e))
        else:
            st.success(f"Estimated insurance charge: ${charge:,.2f}")


main()
