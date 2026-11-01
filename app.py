import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")


# ---- Step 2.a: load the saved model bundle (cached, loaded only once) ----
@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
classes = bundle["classes"]
stats = bundle["stats"]  # {feature: (min, max, mean)}


# ---- Step 2.b: page title and description ----
st.title("🌸 Iris Species Prediction")
st.write(
    "Set the flower measurements below and click **Predict**. "
    "The model is a Random Forest trained on the classic Iris dataset "
    "and predicts one of three species: setosa, versicolor or virginica."
)


# ---- Step 2.c: one slider per feature, limited to the range seen in training ----
st.subheader("Input features")
values = {}
cols = st.columns(len(features))
for col, f in zip(cols, features):
    lo, hi, mean = stats[f]
    values[f] = col.slider(
        f,
        min_value=round(lo, 1),
        max_value=round(hi, 1),
        value=round(mean, 1),
        step=0.1,
    )

input_df = pd.DataFrame([values])
st.subheader("Your input")
st.dataframe(input_df.style.format("{:.1f}"), hide_index=True)


# ---- Step 2.d: run the model and show the prediction ----
if st.button("Predict", type="primary"):
    proba = model.predict_proba(input_df)[0]
    best = int(proba.argmax())
    left, right = st.columns([1, 2])
    with left:
        st.success(f"Predicted species: **{classes[best].capitalize()}**")
        st.metric("Confidence", f"{proba[best]:.1%}")
    with right:
        st.caption("Probability of each species")
        st.bar_chart(
            pd.DataFrame(
                {"Probability": proba},
                index=[c.capitalize() for c in classes],
            ),
            height=220,
        )
