from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="centered"
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_mvp_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "app" / "feature_config.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


def normalize_category_value(value):
    """Convert category value into a clean string key."""
    try:
        numeric_value = float(value)

        if numeric_value.is_integer():
            return str(int(numeric_value))

        return str(numeric_value)
    except (ValueError, TypeError):
        return str(value)


def get_fallback_label(feature):
    """Fallback label for encoded values not listed in feature_config.json."""
    fallback_labels = {
        "Application mode": "Other admission pathway",
        "Course": "Other study program",
        "Mother's qualification": "Other education level",
        "Mother's occupation": "Other occupation",
    }

    return fallback_labels.get(feature, "Other")


def build_display_options(feature, raw_options, value_mappings):
    """Build display labels while keeping original encoded values for the model."""
    feature_mapping = value_mappings.get(feature, {})
    display_to_value = {}

    for value in raw_options:
        normalized_value = normalize_category_value(value)

        display_label = feature_mapping.get(
            normalized_value,
            get_fallback_label(feature)
        )

        # Keep display options clean.
        # If multiple encoded values share the same display label, show it once.
        if display_label not in display_to_value:
            display_to_value[display_label] = value

    return display_to_value


def get_feature_label(feature, feature_labels):
    return feature_labels.get(feature, feature)


def get_feature_description(feature, feature_descriptions):
    return feature_descriptions.get(
        feature,
        "Input feature used by the prediction model."
    )


def get_numeric_config(feature, numeric_inputs, df_processed):
    """Get min, max, default, and step for numerical input."""
    config = numeric_inputs.get(feature, {})

    min_value = config.get("min_value", float(df_processed[feature].min()))
    max_value = config.get("max_value", float(df_processed[feature].max()))
    default_value = config.get("default_value", float(df_processed[feature].mean()))
    step = config.get("step", 1.0)

    return min_value, max_value, default_value, step


model = load_model()
metadata = load_json(METADATA_PATH)
feature_config = load_json(FEATURE_CONFIG_PATH)
df_processed = load_processed_data()


metadata_features = metadata.get("mvp_features", [])
config_features = feature_config.get("features", metadata_features)

feature_labels = feature_config.get("feature_labels", {})
feature_descriptions = feature_config.get("feature_descriptions", {})
value_mappings = feature_config.get("value_mappings", {})
numeric_inputs = feature_config.get("numeric_inputs", {})

continuous_features = metadata.get("continuous_features", [])


# Use config feature order, but validate it against the saved model metadata.
missing_from_metadata = [
    feature for feature in config_features
    if feature not in metadata_features
]

missing_from_config = [
    feature for feature in metadata_features
    if feature not in config_features
]

if missing_from_metadata or missing_from_config:
    st.error("Feature configuration does not match the saved model metadata.")

    with st.expander("Show feature mismatch details"):
        st.write("Missing from model metadata:", missing_from_metadata)
        st.write("Missing from app config:", missing_from_config)

    st.stop()


mvp_features = config_features


st.title("🎓 Student Dropout Prediction")

st.write(
    "This MVP predicts whether a student is more likely to **Graduate** or **Dropout** "
    "using selected early non-academic features."
)

st.info(
    "This prediction is intended as an early warning support tool, "
    "not as an automatic academic decision."
)


st.subheader("Student Information")

st.caption(
    "Categorical inputs are shown as text, but the original encoded values are still sent to the model."
)


user_input = {}
display_input = {}

with st.form("prediction_form"):
    for feature in mvp_features:
        label = get_feature_label(feature, feature_labels)
        help_text = get_feature_description(feature, feature_descriptions)

        if feature in continuous_features:
            min_value, max_value, default_value, step = get_numeric_config(
                feature,
                numeric_inputs,
                df_processed
            )

            if feature == "Age at enrollment":
                selected_value = st.number_input(
                    label=label,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=int(step),
                    help=help_text
                )
            else:
                selected_value = st.number_input(
                    label=label,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(default_value),
                    step=float(step),
                    help=help_text
                )

            user_input[feature] = selected_value
            display_input[label] = selected_value

        else:
            raw_options = sorted(df_processed[feature].dropna().unique().tolist())

            display_to_value = build_display_options(
                feature,
                raw_options,
                value_mappings
            )

            selected_display = st.selectbox(
                label=label,
                options=list(display_to_value.keys()),
                help=help_text
            )

            selected_encoded_value = display_to_value[selected_display]

            user_input[feature] = selected_encoded_value
            display_input[label] = selected_display

    submitted = st.form_submit_button("Predict")


if submitted:
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    prediction_probability = model.predict_proba(input_df)[0]

    graduate_probability = prediction_probability[0]
    dropout_probability = prediction_probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Dropout Risk")
    else:
        st.success("Prediction: Likely to Graduate")

    st.write("### Prediction Probability")
    st.write(f"Graduate Probability: **{graduate_probability:.2%}**")
    st.write(f"Dropout Probability: **{dropout_probability:.2%}**")

    st.progress(float(dropout_probability))

    with st.expander("Show input summary"):
        st.dataframe(pd.DataFrame([display_input]))

    with st.expander("Show encoded input sent to model"):
        st.dataframe(input_df)