"""Evaluate the classifier using the project's held-out test split."""

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_preprocessing import DATA_PATH, PROJECT_ROOT, TARGET_COLUMN, load_data, preprocess_data
from .train import MODEL_PATH


def evaluate_model(model=None, data_path=DATA_PATH, model_path=MODEL_PATH, target_column=TARGET_COLUMN):
    """Return accuracy, per-class metrics, and confusion matrix for the test set."""
    if model is None:
        resolved_model_path = _resolve_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {resolved_model_path}. Run train.py first."
            )
        model = joblib.load(resolved_model_path)

    df = load_data(data_path)
    _, X_test, _, y_test = preprocess_data(df, target_column)
    predictions = model.predict(X_test)
    labels = sorted(y_test.unique())

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
        "labels": labels,
        "y_true": y_test,
        "predictions": predictions,
    }


def _resolve_path(path):
    from pathlib import Path

    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    result = evaluate_model()
    print(f"Accuracy: {result['accuracy']:.2%}")
    print("Per-class metrics:")
    for label in result["labels"]:
        scores = result["report"][label]
        print(
            f"  {label}: precision={scores['precision']:.2f}, "
            f"recall={scores['recall']:.2f}, f1={scores['f1-score']:.2f}"
        )
