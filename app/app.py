from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon=":mortar_board:",
    layout="wide"
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_mvp_model.pkl"
MODEL_PIPELINES_PATH = PROJECT_ROOT / "models" / "model_pipelines.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "app" / "feature_config.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@st.cache_resource
def load_model_pipelines():
    if MODEL_PIPELINES_PATH.exists():
        return joblib.load(MODEL_PIPELINES_PATH)

    return {"Final Model": joblib.load(MODEL_PATH)}


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_report_csv(filename):
    path = REPORTS_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)


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
        "Course": "Other study program",
        "Mother's qualification": "Other education level",
        "Father's qualification": "Other education level",
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


def format_model_option(model_name, best_model_name):
    if model_name == best_model_name:
        return f"{model_name} (Best Model)"

    return model_name


def show_image_if_exists(filename, caption):
    path = FIGURES_DIR / filename

    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing report figure: {filename}")


def show_dataframe_if_exists(filename):
    df = load_report_csv(filename)

    if df is None:
        st.warning(f"Missing report table: {filename}")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_prediction_inputs(
    mvp_features,
    continuous_features,
    feature_labels,
    feature_descriptions,
    value_mappings,
    numeric_inputs,
    df_processed
):
    user_input = {}
    display_input = {}

    left_col, right_col = st.columns(2)

    for index, feature in enumerate(mvp_features):
        target_col = left_col if index % 2 == 0 else right_col
        label = get_feature_label(feature, feature_labels)
        help_text = get_feature_description(feature, feature_descriptions)

        with target_col:
            if feature in continuous_features:
                min_value, max_value, default_value, step = get_numeric_config(
                    feature,
                    numeric_inputs,
                    df_processed
                )

                selected_value = st.number_input(
                    label=label,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=int(step),
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

                user_input[feature] = display_to_value[selected_display]
                display_input[label] = selected_display

            st.caption(help_text)

    return user_input, display_input


def render_prediction_result(selected_model_name, selected_model, input_df, display_input):
    prediction = selected_model.predict(input_df)[0]
    prediction_probability = selected_model.predict_proba(input_df)[0]
    class_probability = {
        int(class_value): float(prediction_probability[index])
        for index, class_value in enumerate(selected_model.classes_)
    }

    graduate_probability = class_probability.get(0, 0.0)
    dropout_probability = class_probability.get(1, 0.0)

    st.subheader("Prediction Result")
    st.caption(f"Model used: {selected_model_name}")

    if int(prediction) == 1:
        st.error("Prediction: Dropout Risk")
    else:
        st.success("Prediction: Likely to Graduate")

    prob_col_1, prob_col_2 = st.columns(2)
    prob_col_1.metric("Graduate confidence", f"{graduate_probability:.2%}")
    prob_col_2.metric("Dropout confidence", f"{dropout_probability:.2%}")

    st.progress(float(dropout_probability), text="Dropout probability")

    with st.expander("Input summary", expanded=False):
        st.dataframe(pd.DataFrame([display_input]), use_container_width=True, hide_index=True)

    with st.expander("Encoded input sent to model", expanded=False):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


def render_report_section(section):
    if section == "Model Comparison":
        st.subheader("Model Comparison")
        st.write(
            "Logistic Regression is the best model in the current run. It has stronger "
            "validation F1-score and ROC-AUC than Random Forest, although Random Forest "
            "has slightly higher Dropout recall on the validation split."
        )

        with st.expander("Validation metrics table", expanded=True):
            show_dataframe_if_exists("validation_model_comparison.csv")

        with st.expander("Validation metrics plot", expanded=False):
            show_image_if_exists(
                "validation_metrics_comparison.png",
                "Validation metric comparison for Logistic Regression and Random Forest."
            )

        with st.expander("Final test metrics", expanded=False):
            show_dataframe_if_exists("final_model_evaluation.csv")

    elif section == "Confusion Matrix":
        st.subheader("Confusion Matrix")
        st.write(
            "The confusion matrices show the balance between correctly identified Graduate "
            "and Dropout records. Logistic Regression is more conservative on Dropout than "
            "Random Forest in validation, while the final test matrix shows the best model's "
            "final error pattern."
        )

        with st.expander("Validation - Logistic Regression", expanded=False):
            show_image_if_exists(
                "validation_confusion_matrix_logistic_regression.png",
                "Validation confusion matrix for Logistic Regression."
            )

        with st.expander("Validation - Random Forest", expanded=False):
            show_image_if_exists(
                "validation_confusion_matrix_random_forest.png",
                "Validation confusion matrix for Random Forest."
            )

        with st.expander("Final test confusion matrix", expanded=True):
            show_image_if_exists(
                "final_test_confusion_matrix.png",
                "Final test confusion matrix for the selected best model."
            )

        with st.expander("Validation classification report", expanded=False):
            show_dataframe_if_exists("validation_classification_report.csv")

    elif section == "ROC Curves":
        st.subheader("ROC Curves")
        st.write(
            "ROC-AUC measures how well the model separates Graduate and Dropout across "
            "thresholds. Logistic Regression has the stronger validation ROC-AUC in the "
            "current experiment."
        )

        with st.expander("Validation ROC comparison", expanded=True):
            show_image_if_exists(
                "validation_roc_curve_comparison.png",
                "Validation ROC curve comparison."
            )

        with st.expander("Final test ROC curve", expanded=False):
            show_image_if_exists(
                "final_test_roc_curve.png",
                "Final test ROC curve for the selected best model."
            )

    elif section == "Feature Importance":
        st.subheader("Feature Importance")
        st.write(
            "Permutation importance is used only for interpretation. The strongest signals "
            "in the final model are Course, Age at enrollment, and Gender. The fixed MVP "
            "feature set is not changed by this importance ranking."
        )

        with st.expander("Final permutation importance table", expanded=True):
            show_dataframe_if_exists("final_permutation_feature_importance.csv")

        with st.expander("Final permutation importance plot", expanded=False):
            show_image_if_exists(
                "final_permutation_feature_importance.png",
                "Final permutation feature importance for the selected best model."
            )

        with st.expander("Validation importance - Logistic Regression", expanded=False):
            show_image_if_exists(
                "validation_permutation_importance_logistic_regression.png",
                "Validation permutation importance for Logistic Regression."
            )

        with st.expander("Validation importance - Random Forest", expanded=False):
            show_image_if_exists(
                "validation_permutation_importance_random_forest.png",
                "Validation permutation importance for Random Forest."
            )


models = load_model_pipelines()
metadata = load_json(METADATA_PATH)
feature_config = load_json(FEATURE_CONFIG_PATH)
df_processed = load_processed_data()

metadata_features = metadata.get("mvp_features", [])
config_features = feature_config.get("features", metadata_features)
best_model_name = metadata.get("best_model", metadata.get("base_model", ""))
available_models = [
    model_name for model_name in metadata.get("available_models", list(models.keys()))
    if model_name in models
]

feature_labels = feature_config.get("feature_labels", {})
feature_descriptions = feature_config.get("feature_descriptions", {})
value_mappings = feature_config.get("value_mappings", {})
numeric_inputs = feature_config.get("numeric_inputs", {})
continuous_features = metadata.get("continuous_features", [])

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

if not available_models:
    st.error("No selectable model pipelines are available.")
    st.stop()

mvp_features = config_features

st.title("Student Dropout Prediction")
st.write(
    "This MVP predicts whether a student is more likely to Graduate or Dropout "
    "using selected early enrollment and background features."
)
st.info(
    "This prediction is intended as an early warning support tool, "
    "not as an automatic academic decision."
)

prediction_tab, report_tab = st.tabs(["Prediction", "Reports"])

with prediction_tab:
    st.subheader("Student Information")
    st.caption(
        "Categorical inputs are shown as readable labels. The model still receives "
        "the original encoded values behind the scenes."
    )

    model_display_options = {
        format_model_option(model_name, best_model_name): model_name
        for model_name in available_models
    }

    selected_model_display = st.selectbox(
        "Model",
        options=list(model_display_options.keys()),
        help="Choose which trained model should make the prediction."
    )
    selected_model_name = model_display_options[selected_model_display]
    selected_model = models[selected_model_name]

    with st.form("prediction_form"):
        user_input, display_input = render_prediction_inputs(
            mvp_features,
            continuous_features,
            feature_labels,
            feature_descriptions,
            value_mappings,
            numeric_inputs,
            df_processed
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        render_prediction_result(
            selected_model_name,
            selected_model,
            input_df,
            display_input
        )

with report_tab:
    st.subheader("Model Reports")
    st.caption(
        "Choose one report segment at a time. Each segment contains collapsible tables "
        "and plots so the page stays easier to scan."
    )

    selected_report_section = st.radio(
        "Report segment",
        options=[
            "Model Comparison",
            "Confusion Matrix",
            "ROC Curves",
            "Feature Importance"
        ],
        horizontal=True
    )

    render_report_section(selected_report_section)
