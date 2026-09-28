"""Streamlit dashboard for training, evaluating, and using the Iris classifier."""

import pandas as pd
import streamlit as st

from src.data_preprocessing import DATA_PATH, TARGET_COLUMN, load_data
from src.evaluate import evaluate_model
from src.predict import predict_species
from src.train import MODEL_PATH, train_model


st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌼",
    layout="wide",
)


@st.cache_data
def get_dataset():
    return load_data(DATA_PATH)


@st.cache_resource
def get_model():
    return train_model(DATA_PATH, MODEL_PATH, TARGET_COLUMN)


st.title("🌼 Iris Species Classifier")
st.caption("Explore the dataset, review model performance, and classify a flower.")

try:
    dataset = get_dataset()
    model = get_model()
    evaluation = evaluate_model(model=model, data_path=DATA_PATH)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not prepare the classifier: {error}")
    st.stop()

overview_tab, predict_tab, data_tab = st.tabs(["Overview", "Predict", "Dataset"])

with overview_tab:
    score_col, row_col, feature_col, model_col = st.columns(4)
    score_col.metric("Test accuracy", f"{evaluation['accuracy']:.1%}")
    row_col.metric("Dataset rows", f"{len(dataset):,}")
    feature_col.metric("Input features", f"{len(dataset.columns) - 1}")
    model_col.metric("Model", "Gaussian Naive Bayes")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Confusion matrix")
        matrix = pd.DataFrame(
            evaluation["confusion_matrix"],
            index=evaluation["labels"],
            columns=evaluation["labels"],
        )
        st.dataframe(matrix, width="stretch")
    with right:
        st.subheader("Per species performance")
        report = pd.DataFrame(evaluation["report"]).T
        st.dataframe(
            report.loc[evaluation["labels"], ["precision", "recall", "f1-score"]]
            .rename(columns={"f1-score": "f1 score"})
            .style.format("{:.2f}"),
            width="stretch",
        )

with predict_tab:
    st.subheader("Enter flower measurements")
    st.write("Measurements are in centimeters. The starting values use dataset averages.")

    feature_columns = [column for column in dataset.columns if column != TARGET_COLUMN]
    with st.form("prediction_form"):
        inputs = {}
        fields = st.columns(2)
        for index, column in enumerate(feature_columns):
            values = dataset[column]
            with fields[index % 2]:
                inputs[column] = st.number_input(
                    column.replace("_", " ").title(),
                    min_value=float(values.min()),
                    max_value=float(values.max()),
                    value=float(values.mean()),
                    step=0.1,
                    format="%.1f",
                )
        submitted = st.form_submit_button("Predict species", type="primary")

    if submitted:
        result = predict_species(inputs, model=model)
        st.success(f"Predicted species: **{result['species']}**")
        st.caption(f"Model confidence: {result['confidence']:.1%}")
        probabilities = pd.DataFrame(
            {"Species": result["probabilities"].keys(), "Probability": result["probabilities"].values()}
        ).set_index("Species")
        st.bar_chart(probabilities)

with data_tab:
    st.subheader("Iris dataset")
    st.write(f"Target column: **{TARGET_COLUMN}**")
    st.dataframe(dataset, width="stretch", hide_index=True)
    st.download_button(
        "Download dataset as CSV",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name=DATA_PATH.name,
        mime="text/csv",
    )

st.sidebar.header("Model")
st.sidebar.write("Gaussian Naive Bayes")
st.sidebar.caption(f"Saved model: `{MODEL_PATH.relative_to(MODEL_PATH.parents[1])}`")
if st.sidebar.button("Retrain model"):
    get_model.clear()
    st.rerun()
