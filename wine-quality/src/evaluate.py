import json
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
from src.utils import read_yaml, ensure_dir

def main():
    params = read_yaml('params.yaml')
    target_col = params['data']['target']
    avg = params['evaluate']['average']

    test = pd.read_csv('data/processed/test.csv')
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    model = joblib.load('artifacts/model/model.pkl')

    # Predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average=avg)

    ensure_dir('artifacts/metrics')
    metrics = {'test_accuracy': acc, 'test_f1_macro': f1m}
    with open('artifacts/metrics/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    fig.savefig('artifacts/metrics/confusion_matrix.png', dpi=150)
    plt.close(fig)

    # Log to MLflow
    mlflow.set_experiment('wine-quality-experiment')
    with mlflow.start_run(run_name='evaluation'):
        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric('test_f1_macro', f1m)
        mlflow.log_artifact('artifacts/metrics/metrics.json')
        mlflow.log_artifact('artifacts/metrics/confusion_matrix.png')

    print('Saved metrics and confusion matrix.')

if __name__ == '__main__':
    main()