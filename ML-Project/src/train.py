"""Train and persist the Iris species classifier."""

from pathlib import Path

import joblib
from sklearn.naive_bayes import GaussianNB

from .data_preprocessing import DATA_PATH, PROJECT_ROOT, TARGET_COLUMN, load_data, preprocess_data


MODEL_PATH = PROJECT_ROOT / "models" / "iris_classifier.joblib"


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH, target_column=TARGET_COLUMN):
    """Fit a scaled logistic regression model and save it to ``model_path``."""
    df = load_data(data_path)
    X_train, _, y_train, _ = preprocess_data(df, target_column)

    model = GaussianNB()
    model.fit(X_train, y_train)

    output_path = _resolve_model_path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def _resolve_model_path(model_path):
    path = Path(model_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    trained_model = train_model()
    print(f"Trained {trained_model.__class__.__name__} and saved it to {MODEL_PATH}.")
