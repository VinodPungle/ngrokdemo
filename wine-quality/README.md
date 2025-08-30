# Wine Quality – MLOps (Git + DVC + MLflow + Streamlit)

This project predicts wine quality (multiclass classification) from the classic `winequality-red.csv` dataset.
It includes:
- **Data versioning** with DVC
- **Experiment tracking** with MLflow
- **A Streamlit app** for interactive inference
- A simple, reproducible pipeline managed by **`dvc.yaml`** and **`params.yaml`**

## 0) Prereqs
- Windows laptop
- Git Bash (to run Git/DVC commands)
- Python 3.10+ in PATH (`python --version`)
- Git + DVC installed
- Repo path (local): `C:\Users\Vinod\ngrokdemo` (Git Bash path: `/c/Users/Vinod/ngrokdemo`)

## 1) Put this project in your repo
Copy the entire `wine-quality-mlops/` folder into your repo (e.g., inside `ngrokdemo/`).  
You could rename to just `wine-quality/` if you like.

### From Git Bash
```bash
cd /c/Users/Vinod/ngrokdemo
# If you want a subfolder named 'wine-quality', move it:
mv /mnt/data/wine-quality-mlops ./wine-quality
cd wine-quality
```

## 2) Create & activate a virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # (Git Bash on Windows)
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Initialize Git (if not already) and DVC
```bash
# If repo already initialized & remote set, skip init
git status

# Initialize DVC (do this inside the wine-quality folder)
dvc init
git add .dvc .dvcignore
git commit -m "init dvc"
```

## 4) Track the dataset with DVC
The raw CSV is at `data/raw/winequality-red.csv` (included here for convenience). Track it with DVC:
```bash
dvc add data/raw/winequality-red.csv
git add data/raw/winequality-red.csv.dvc .gitignore
git commit -m "track dataset with dvc"
```

> Optional (recommended): add a DVC remote (local folder for now). You can switch to cloud later.
```bash
dvc remote add -d localremote /c/Users/Vinod/dvcstore
dvc push  # uploads data to the DVC remote
git add .dvc/config
git commit -m "configure local dvc remote"
```

## 5) Run the pipeline (DVC)
```bash
# 1) Split
dvc repro split
# 2) Train
dvc repro train
# 3) Evaluate
dvc repro evaluate
# …or just run all stages:
dvc repro
```

Outputs:
- Model: `artifacts/model/model.pkl`
- Metrics: `artifacts/metrics/metrics.json`
- Confusion matrix: `artifacts/metrics/confusion_matrix.png`

## 6) Track experiments with MLflow
The training and evaluation scripts log to MLflow automatically (local `./mlruns` directory).
```bash
mlflow ui
# Open http://127.0.0.1:5000 in your browser
```

## 7) Commit & push
```bash
git add -A
git commit -m "wine quality pipeline (dvc + mlflow + streamlit)"
git push origin main  # or your branch
```

## 8) Serve the model with Streamlit
```bash
# Make sure a model exists (run the pipeline first)
streamlit run streamlit_app.py
```

## 9) Tweaking
- Change hyperparams in `params.yaml`, then: `dvc repro`
- Switch model type in `params.yaml` (e.g., `LogisticRegression`) and adjust code if needed
- To use a cloud DVC remote later (e.g., Google Drive, S3), run `dvc remote add` with that URI

---
### Project layout
```text
.
├── artifacts/
│   ├── metrics/
│   └── model/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_prep.py
│   ├── train.py
│   └── evaluate.py
├── streamlit_app.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .gitignore
└── .dvcignore
```