from html import escape
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="EarlyDrop | Student Risk Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_mvp_model.pkl"
MODEL_PIPELINES_PATH = PROJECT_ROOT / "models" / "model_pipelines.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "app" / "feature_config.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

GITHUB_URL = "https://github.com/LecyLecy/student-dropout-prediction-ml"
DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/697/"
    "predict+students+dropout+and+academic+success"
)


APP_STYLES = """
<style>
    :root {
        --ink: #132238;
        --muted: #5f6f82;
        --line: #dfe6ee;
        --surface: #ffffff;
        --canvas: #f4f7fb;
        --navy: #183b56;
        --teal: #087f7a;
        --teal-soft: #e8f6f3;
        --amber: #b96713;
        --amber-soft: #fff5e7;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .brand-row {
        align-items: center;
        display: flex;
        gap: 0.75rem;
        margin-bottom: 2rem;
    }

    .brand-mark {
        align-items: center;
        background: var(--navy);
        border-radius: 11px;
        color: white;
        display: flex;
        font-size: 0.9rem;
        font-weight: 800;
        height: 38px;
        justify-content: center;
        letter-spacing: -0.03em;
        width: 38px;
    }

    .brand-name {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: -0.02em;
    }

    .brand-tagline {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: -0.1rem;
    }

    .hero {
        background:
            radial-gradient(circle at 92% 15%, rgba(47, 169, 158, 0.22), transparent 28%),
            linear-gradient(135deg, #142f46 0%, #1d4b62 58%, #176b69 100%);
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(27, 57, 78, 0.13);
        color: white;
        margin-bottom: 1.6rem;
        overflow: hidden;
        padding: 2.7rem 3rem;
        position: relative;
    }

    .hero-kicker {
        color: #87ddd4;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3.45rem);
        letter-spacing: -0.045em;
        line-height: 1.04;
        margin: 0;
        max-width: 760px;
    }

    .hero p {
        color: #dbe9ee;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 1.15rem 0 0;
        max-width: 720px;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.5rem;
    }

    .hero-badge {
        background: rgba(255, 255, 255, 0.11);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 100px;
        color: #f5fbfc;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.48rem 0.8rem;
    }

    .section-heading {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 760;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 1.15rem;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        min-height: 122px;
        padding: 1.25rem 1.3rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--ink);
        font-size: 1.85rem;
        font-weight: 780;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-top: 0.4rem;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.76rem;
        margin-top: 0.3rem;
    }

    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(21, 48, 68, 0.05);
        padding: 1.4rem 1.5rem 1.55rem;
    }

    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        gap: 1.15rem;
    }

    div[data-baseweb="select"] > div,
    [data-testid="stNumberInputContainer"] {
        border-color: #d7e0e9;
        border-radius: 10px;
        min-height: 45px;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 720;
        min-height: 46px;
    }

    [data-testid="stFormSubmitButton"] > button {
        background: var(--teal);
        border: 1px solid var(--teal);
        box-shadow: 0 8px 18px rgba(8, 127, 122, 0.16);
        color: white;
        width: 100%;
    }

    [data-testid="stFormSubmitButton"] > button:hover {
        background: #066e6a;
        border-color: #066e6a;
        color: white;
    }

    button[data-baseweb="tab"] {
        font-weight: 700;
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.7rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--teal);
    }

    .risk-card {
        border: 1px solid;
        border-radius: 20px;
        margin-top: 1.2rem;
        padding: 1.5rem 1.6rem;
    }

    .risk-card.elevated {
        background: var(--amber-soft);
        border-color: #f0c78d;
    }

    .risk-card.lower {
        background: var(--teal-soft);
        border-color: #a8dcd4;
    }

    .risk-status {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 780;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .risk-summary {
        color: var(--muted);
        line-height: 1.6;
        margin: 0;
    }

    .risk-probability {
        color: var(--ink);
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1;
        margin: 0.85rem 0 0.15rem;
    }

    .risk-probability-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 650;
    }

    .note-card {
        background: #eef4f8;
        border-left: 4px solid #52758b;
        border-radius: 0 12px 12px 0;
        color: #415a6b;
        font-size: 0.86rem;
        line-height: 1.6;
        margin: 1rem 0;
        padding: 0.9rem 1rem;
    }

    .workflow {
        display: grid;
        gap: 0.8rem;
        grid-template-columns: repeat(4, 1fr);
        margin: 1rem 0 1.6rem;
    }

    .workflow-step {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem;
    }

    .workflow-number {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 800;
    }

    .workflow-title {
        color: var(--ink);
        font-size: 0.9rem;
        font-weight: 720;
        margin-top: 0.3rem;
    }

    .workflow-copy {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
        margin-top: 0.25rem;
    }

    .footer {
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding: 1.2rem 0 0.5rem;
        text-align: center;
    }

    .footer a {
        color: var(--teal);
        font-weight: 700;
        text-decoration: none;
    }

    @media (max-width: 740px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem;
        }

        .hero {
            border-radius: 18px;
            padding: 2rem 1.35rem;
        }

        .hero h1 {
            font-size: 2.15rem;
        }

        .workflow {
            grid-template-columns: 1fr 1fr;
        }
    }
</style>
"""


@st.cache_resource
def load_model_pipelines():
    try:
        if MODEL_PIPELINES_PATH.exists():
            return joblib.load(MODEL_PIPELINES_PATH)

        return {"Final Model": joblib.load(MODEL_PATH)}
    except ModuleNotFoundError as error:
        st.error(
            "The saved model could not be loaded because a required Python "
            f"package is missing: `{error.name}`. Install the pinned "
            "dependencies from `requirements.txt` and redeploy the app."
        )
        st.stop()


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
    return pd.read_csv(path) if path.exists() else None


def normalize_category_value(value):
    """Convert a category value into a clean string key."""
    try:
        numeric_value = float(value)
        return str(int(numeric_value)) if numeric_value.is_integer() else str(numeric_value)
    except (ValueError, TypeError):
        return str(value)


def get_fallback_label(feature):
    fallback_labels = {
        "Course": "Other study program",
        "Mother's qualification": "Other education level",
        "Father's qualification": "Other education level",
    }
    return fallback_labels.get(feature, "Other")


def build_display_options(feature, raw_options, value_mappings):
    """Keep model values encoded while presenting readable English labels."""
    feature_mapping = value_mappings.get(feature, {})
    display_to_value = {}

    for value in raw_options:
        normalized_value = normalize_category_value(value)
        display_label = feature_mapping.get(
            normalized_value,
            get_fallback_label(feature),
        )
        if display_label not in display_to_value:
            display_to_value[display_label] = value

    return display_to_value


def get_numeric_config(feature, numeric_inputs, df_processed):
    config = numeric_inputs.get(feature, {})
    return (
        config.get("min_value", float(df_processed[feature].min())),
        config.get("max_value", float(df_processed[feature].max())),
        config.get("default_value", float(df_processed[feature].mean())),
        config.get("step", 1.0),
    )


def show_image_if_exists(filename, caption):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Report figure unavailable: {filename}")


def render_brand():
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">ED</div>
            <div>
                <div class="brand-name">EarlyDrop</div>
                <div class="brand-tagline">Student success analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Early intervention support</div>
            <h1>Student dropout risk, identified earlier.</h1>
            <p>
                A machine-learning assessment that uses enrollment and background
                information to help student-support teams prioritize timely,
                human-led outreach.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">10 early-stage factors</span>
                <span class="hero-badge">No semester-grade leakage</span>
                <span class="hero-badge">Recall-first evaluation</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(eyebrow, title, copy):
    st.markdown(
        f"""
        <div class="eyebrow">{escape(eyebrow)}</div>
        <div class="section-heading">{escape(title)}</div>
        <div class="section-copy">{escape(copy)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_input(
    feature,
    feature_labels,
    feature_descriptions,
    value_mappings,
    numeric_inputs,
    continuous_features,
    df_processed,
):
    label = feature_labels.get(feature, feature)
    help_text = feature_descriptions.get(
        feature,
        "Input factor used by the prediction model.",
    )

    if feature in continuous_features:
        min_value, max_value, default_value, step = get_numeric_config(
            feature,
            numeric_inputs,
            df_processed,
        )
        selected_value = st.number_input(
            label=label,
            min_value=int(min_value),
            max_value=int(max_value),
            value=int(default_value),
            step=int(step),
            help=help_text,
            key=f"input_{feature}",
        )
        return selected_value, selected_value

    raw_options = sorted(df_processed[feature].dropna().unique().tolist())
    display_to_value = build_display_options(
        feature,
        raw_options,
        value_mappings,
    )
    selected_display = st.selectbox(
        label=label,
        options=list(display_to_value.keys()),
        help=help_text,
        key=f"input_{feature}",
    )
    return display_to_value[selected_display], selected_display


def render_input_group(
    title,
    features,
    user_input,
    display_input,
    feature_labels,
    feature_descriptions,
    value_mappings,
    numeric_inputs,
    continuous_features,
    df_processed,
):
    st.markdown(f"**{title}**")
    columns = st.columns(2)

    for index, feature in enumerate(features):
        with columns[index % 2]:
            model_value, display_value = render_feature_input(
                feature,
                feature_labels,
                feature_descriptions,
                value_mappings,
                numeric_inputs,
                continuous_features,
                df_processed,
            )
            user_input[feature] = model_value
            display_input[feature_labels.get(feature, feature)] = display_value


def render_assessment_form(
    feature_labels,
    feature_descriptions,
    value_mappings,
    numeric_inputs,
    continuous_features,
    df_processed,
):
    user_input = {}
    display_input = {}

    input_groups = [
        (
            "Student profile",
            ["Age at enrollment", "Gender", "Marital status", "International"],
        ),
        (
            "Academic background",
            ["Course", "Previous qualification"],
        ),
        (
            "Family education",
            ["Mother's qualification", "Father's qualification"],
        ),
        (
            "Access and support",
            ["Displaced", "Educational special needs"],
        ),
    ]

    with st.form("risk_assessment_form"):
        for group_index, (title, features) in enumerate(input_groups):
            if group_index:
                st.divider()
            render_input_group(
                title,
                features,
                user_input,
                display_input,
                feature_labels,
                feature_descriptions,
                value_mappings,
                numeric_inputs,
                continuous_features,
                df_processed,
            )

        st.write("")
        submitted = st.form_submit_button(
            "Assess student risk",
            type="primary",
            width="stretch",
        )

    return submitted, user_input, display_input


def render_prediction_result(
    selected_model_name,
    selected_model,
    threshold,
    input_df,
    display_input,
):
    prediction_probability = selected_model.predict_proba(input_df)[0]
    class_probability = {
        int(class_value): float(prediction_probability[index])
        for index, class_value in enumerate(selected_model.classes_)
    }

    graduation_probability = class_probability.get(0, 0.0)
    dropout_probability = class_probability.get(1, 0.0)
    is_elevated = dropout_probability >= threshold

    status = "Elevated risk — review recommended" if is_elevated else "Lower risk — routine support"
    summary = (
        "This profile crossed the model's review threshold. A student-support "
        "professional should consider the wider context before deciding on outreach."
        if is_elevated
        else
        "This profile did not cross the model's review threshold. Continue normal "
        "support and monitor new information as it becomes available."
    )
    card_class = "elevated" if is_elevated else "lower"

    st.markdown(
        f"""
        <div class="risk-card {card_class}">
            <div class="eyebrow">Assessment result</div>
            <div class="risk-status">{escape(status)}</div>
            <p class="risk-summary">{escape(summary)}</p>
            <div class="risk-probability">{dropout_probability:.1%}</div>
            <div class="risk-probability-label">Estimated dropout probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        float(dropout_probability),
        text=f"Review threshold: {threshold:.0%}",
    )

    probability_col, graduation_col, model_col = st.columns(3)
    probability_col.metric("Dropout probability", f"{dropout_probability:.1%}")
    graduation_col.metric("Graduation probability", f"{graduation_probability:.1%}")
    model_col.metric("Model", selected_model_name)

    st.markdown(
        """
        <div class="note-card">
            This output is a screening signal, not a diagnosis or academic decision.
            It should never be used as the sole reason to deny access, change
            enrollment status, or take disciplinary action.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Review the submitted profile"):
        summary_df = pd.DataFrame(
            {
                "Input": list(display_input.keys()),
                "Selected value": [
                    str(value) for value in display_input.values()
                ],
            }
        )
        st.dataframe(summary_df, width="stretch", hide_index=True)


def render_assessment_tab(
    models,
    available_models,
    best_model_name,
    model_thresholds,
    feature_labels,
    feature_descriptions,
    value_mappings,
    numeric_inputs,
    continuous_features,
    df_processed,
):
    render_section_intro(
        "Risk assessment",
        "Build an early student profile",
        (
            "Complete the ten fields below. The assessment intentionally excludes "
            "semester grades and post-enrollment financial status."
        ),
    )

    model_display_options = {
        (
            f"{model_name} · recommended"
            if model_name == best_model_name
            else model_name
        ): model_name
        for model_name in available_models
    }
    display_labels = list(model_display_options.keys())
    default_index = next(
        (
            index
            for index, label in enumerate(display_labels)
            if model_display_options[label] == best_model_name
        ),
        0,
    )

    with st.expander("Model settings", expanded=False):
        selected_model_display = st.selectbox(
            "Prediction model",
            options=display_labels,
            index=default_index,
            help=(
                "Random Forest is recommended because it delivered the strongest "
                "recall-first result in the final evaluation."
            ),
        )
        selected_model_name = model_display_options[selected_model_display]
        selected_threshold = float(model_thresholds.get(selected_model_name, 0.50))
        st.caption(
            f"{selected_model_name} flags profiles at a "
            f"{selected_threshold:.0%} estimated dropout probability."
        )

    submitted, user_input, display_input = render_assessment_form(
        feature_labels,
        feature_descriptions,
        value_mappings,
        numeric_inputs,
        continuous_features,
        df_processed,
    )

    if submitted:
        selected_model = models[selected_model_name]
        input_df = pd.DataFrame([user_input])
        render_prediction_result(
            selected_model_name,
            selected_model,
            selected_threshold,
            input_df,
            display_input,
        )


def render_performance_tab(metadata):
    evaluation = metadata.get("evaluation", [])
    final_metrics = evaluation[0] if evaluation else {}

    render_section_intro(
        "Model performance",
        "Evaluation built around early detection",
        (
            "The recommended Random Forest uses a tuned 0.40 threshold. This "
            "increases dropout recall so fewer at-risk students are missed."
        ),
    )

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card(
            "Dropout recall",
            f"{final_metrics.get('Recall', 0):.1%}",
            "At-risk cases identified",
        )
    with metric_columns[1]:
        render_metric_card(
            "F1 score",
            f"{final_metrics.get('F1-Score', 0):.3f}",
            "Precision–recall balance",
        )
    with metric_columns[2]:
        render_metric_card(
            "ROC–AUC",
            f"{final_metrics.get('ROC-AUC', 0):.3f}",
            "Ranking performance",
        )
    with metric_columns[3]:
        render_metric_card(
            "Decision threshold",
            f"{final_metrics.get('Threshold', 0):.0%}",
            "Recall-first operating point",
        )

    st.write("")
    benchmark_tab, feature_tab, threshold_tab = st.tabs(
        ["Model benchmark", "Feature signals", "Threshold analysis"]
    )

    with benchmark_tab:
        comparison_df = load_report_csv("validation_model_comparison.csv")
        if comparison_df is not None:
            display_df = comparison_df.copy()
            metric_columns_to_format = [
                "Recall",
                "Precision",
                "F1-Score",
                "ROC-AUC",
            ]
            for column in metric_columns_to_format:
                display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")
            st.dataframe(display_df, width="stretch", hide_index=True)

        show_image_if_exists(
            "validation_metrics_comparison.png",
            "Five-fold cross-validation comparison across candidate models.",
        )

        st.markdown(
            """
            <div class="note-card">
                Selection prioritizes dropout recall and F1 score over raw accuracy.
                In an early-warning workflow, a false alert can be reviewed by a
                person, while a missed at-risk student may receive no timely support.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature_tab:
        show_image_if_exists(
            "final_feature_importance.png",
            "Relative feature importance for the final Random Forest model.",
        )
        st.caption(
            "Feature importance describes model reliance, not causation. Sensitive "
            "background variables require careful monitoring before real-world use."
        )

    with threshold_tab:
        show_image_if_exists(
            "random_forest_threshold_sweep.png",
            "Precision, recall, and F1 trade-offs across Random Forest thresholds.",
        )
        show_image_if_exists(
            "final_test_confusion_matrix.png",
            "Final test-set confusion matrix at the selected 0.40 threshold.",
        )


def render_about_tab(df_processed):
    render_section_intro(
        "Project overview",
        "From raw enrollment data to a deployable screening tool",
        (
            "This portfolio project demonstrates an end-to-end classical machine "
            "learning workflow: exploratory analysis, leakage-aware preprocessing, "
            "model selection, threshold tuning, evaluation, and deployment."
        ),
    )

    project_columns = st.columns(3)
    with project_columns[0]:
        render_metric_card(
            "Dataset",
            f"{len(df_processed):,}",
            "Graduate and dropout records",
        )
    with project_columns[1]:
        render_metric_card(
            "Input scope",
            "10 factors",
            "Available at or near enrollment",
        )
    with project_columns[2]:
        render_metric_card(
            "Models tested",
            "5",
            "Classical classification pipelines",
        )

    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step">
                <div class="workflow-number">01</div>
                <div class="workflow-title">Explore</div>
                <div class="workflow-copy">Inspect distributions, quality, and early feature relationships.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">02</div>
                <div class="workflow-title">Prepare</div>
                <div class="workflow-copy">Remove leakage and apply feature-specific preprocessing.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">03</div>
                <div class="workflow-title">Evaluate</div>
                <div class="workflow-copy">Compare five models with stratified cross-validation.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">04</div>
                <div class="workflow-title">Deploy</div>
                <div class="workflow-copy">Serve the selected pipeline in an accessible Streamlit app.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Modeling choices")
        st.markdown(
            """
            - Binary target: **Graduate (0)** vs. **Dropout (1)**
            - Five-fold stratified cross-validation
            - Feature-specific scaling and categorical encoding
            - Random Forest selected with a recall-first threshold
            - Saved end-to-end pipelines prevent training/serving skew
            """
        )

    with right_column:
        st.subheader("Responsible-use boundaries")
        st.markdown(
            """
            - Human review is required for every flagged profile
            - Predictions should support outreach, never punishment
            - Feature importance does not establish causation
            - Performance should be monitored across student groups
            - Local validation is required before institutional use
            """
        )

    st.info(
        "Portfolio demonstration only. The source dataset reflects one higher-"
        "education context, so the model is not validated for direct operational "
        "use at other institutions."
    )

    link_column, dataset_column = st.columns(2)
    link_column.link_button(
        "View the source code on GitHub",
        GITHUB_URL,
        width="stretch",
    )
    dataset_column.link_button(
        "Explore the UCI dataset",
        DATASET_URL,
        width="stretch",
    )


def validate_feature_configuration(config_features, metadata_features):
    missing_from_metadata = [
        feature for feature in config_features if feature not in metadata_features
    ]
    missing_from_config = [
        feature for feature in metadata_features if feature not in config_features
    ]

    if missing_from_metadata or missing_from_config:
        st.error("The app feature configuration does not match the saved model.")
        with st.expander("Configuration details"):
            st.write("Missing from model metadata:", missing_from_metadata)
            st.write("Missing from app configuration:", missing_from_config)
        st.stop()


def main():
    st.markdown(APP_STYLES, unsafe_allow_html=True)

    models = load_model_pipelines()
    metadata = load_json(METADATA_PATH)
    feature_config = load_json(FEATURE_CONFIG_PATH)
    df_processed = load_processed_data()

    metadata_features = metadata.get("mvp_features", [])
    config_features = feature_config.get("features", metadata_features)
    validate_feature_configuration(config_features, metadata_features)

    best_model_name = metadata.get("best_model", metadata.get("base_model", ""))
    available_models = [
        model_name
        for model_name in metadata.get("available_models", list(models.keys()))
        if model_name in models
    ]
    if not available_models:
        st.error("No prediction model pipelines are available.")
        st.stop()

    feature_groups = metadata.get("feature_groups", {})
    continuous_features = metadata.get(
        "continuous_features",
        feature_groups.get("continuous_features", []),
    )

    render_brand()
    render_hero()

    assessment_tab, performance_tab, about_tab = st.tabs(
        ["Risk assessment", "Model performance", "About the project"]
    )

    with assessment_tab:
        render_assessment_tab(
            models=models,
            available_models=available_models,
            best_model_name=best_model_name,
            model_thresholds=metadata.get("model_thresholds", {}),
            feature_labels=feature_config.get("feature_labels", {}),
            feature_descriptions=feature_config.get("feature_descriptions", {}),
            value_mappings=feature_config.get("value_mappings", {}),
            numeric_inputs=feature_config.get("numeric_inputs", {}),
            continuous_features=continuous_features,
            df_processed=df_processed,
        )

    with performance_tab:
        render_performance_tab(metadata)

    with about_tab:
        render_about_tab(df_processed)

    st.markdown(
        """
        <div class="footer">
            EarlyDrop · Built as an end-to-end machine learning portfolio project
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
