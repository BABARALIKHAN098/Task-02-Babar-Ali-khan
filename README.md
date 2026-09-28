# Iris Species Classifier

Streamlit app to explore the Iris dataset, train a Gaussian Naive Bayes classifier, inspect its held-out evaluation, and predict a species from flower measurements.

## Run locally

From the `ML-Project` directory, install dependencies and start the app:

```powershell
..\venv\python.exe -m pip install -r requirements.txt
..\venv\python.exe -m streamlit run app.py
```

If you use another Python environment, activate it first, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The model is trained on the first app launch and saved at `models/iris_classifier.joblib`.

## Source modules

Run the pipeline modules from this directory in order:

```powershell
..\venv\python.exe -m src.data_preprocessing
..\venv\python.exe -m src.train
..\venv\python.exe -m src.evaluate
..\venv\python.exe -m src.predict
```
