"""Prediction helpers for the Iris species classifier."""

import joblib
import pandas as pd

from .data_preprocessing import PROJECT_ROOT
from .train import MODEL_PATH


def predict_species(features, model=None, model_path=MODEL_PATH):
    """Predict a species from a mapping, one-row DataFrame, or feature sequence."""
    if model is None:
        path = _resolve_model_path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
        model = joblib.load(path)

    feature_names = list(model.feature_names_in_)
    if isinstance(features, pd.DataFrame):
        row = features.loc[:, feature_names]
    elif isinstance(features, dict):
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing feature values: {', '.join(missing)}")
        row = pd.DataFrame([{name: features[name] for name in feature_names}])
    else:
        row = pd.DataFrame([features], columns=feature_names)

    probabilities = model.predict_proba(row)[0]
    class_names = model.classes_
    best_index = probabilities.argmax()
    return {
        "species": class_names[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            label: float(probability)
            for label, probability in zip(class_names, probabilities)
        },
    }


def _resolve_model_path(model_path):
    from pathlib import Path

    path = Path(model_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    from .data_preprocessing import DATA_PATH, load_data

    example = load_data().drop(columns="species").iloc[0].to_dict()
    prediction = predict_species(example)
    print(f"Example prediction: {prediction['species']} ({prediction['confidence']:.1%})")
