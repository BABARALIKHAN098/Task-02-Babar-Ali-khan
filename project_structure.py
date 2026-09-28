
from pathlib import Path

# Project name
project_name = "ML-Project"

# Define project structure
files = [
    "data/dataset.csv",
    "models/model.pkl",
    "src/__init__.py",
    "src/data_preprocessing.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "app.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Create project directory
project_path = Path(project_name)
project_path.mkdir(exist_ok=True)

# Create folders and files
for file in files:
    file_path = project_path / file

    # Create parent folder
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create empty file
    file_path.touch(exist_ok=True)

print("ML project structure created successfully!")

# Display structure
print("\nProject Structure:")
print("""
ML-Project/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
""")

