import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.sklearn

from src.utils import read_yaml, ensure_dir

def build_model(model_cfg: dict):
    mtype = model_cfg.get('type', 'RandomForestClassifier')
    if mtype == 'RandomForestClassifier':
        return RandomForestClassifier(
            n_estimators=model_cfg.get('n_estimators', 200),
            max_depth=model_cfg.get('max_depth', None),
            class_weight=model_cfg.get('class_weight', None),
            random_state=42
        )
    elif mtype == 'LogisticRegression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000))
        ])
    else:
        raise ValueError(f"Unsupported model type: {mtype}")

def main():
    params = read_yaml('params.yaml')
    tracking = params.get('tracking', {})
    experiment_name = tracking.get('experiment_name', 'wine-quality-experiment')
    run_name = tracking.get('run_name', 'rf-baseline')

    # MLflow config
    mlflow.set_experiment(experiment_name)

    # Data
    train = pd.read_csv('data/processed/train.csv')
    target_col = params['data']['target']
    X = train.drop(columns=[target_col])
    y = train[target_col]

    # Model
    model = build_model(params['model'])

    with mlflow.start_run(run_name=run_name):
        # Log hyperparams
        for k, v in params['model'].items():
            mlflow.log_param(k, v)

        # Fit
        model.fit(X, y)

        # Quick train metrics (not ideal for generalization, but useful)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1m = f1_score(y, preds, average=params['evaluate']['average'])
        mlflow.log_metric('train_accuracy', acc)
        mlflow.log_metric('train_f1_macro', f1m)

        # Save artifacts
        ensure_dir('artifacts/model')
        joblib.dump(model, 'artifacts/model/model.pkl')
        # Save feature names for app
        X_cols = list(X.columns)
        import json as _json
        with open('artifacts/model/feature_names.json', 'w', encoding='utf-8') as f:
            _json.dump(X_cols, f, indent=2)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path='model')

    print('Model trained and saved to artifacts/model/model.pkl')

if __name__ == '__main__':
    main()