import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import MODEL_PATH, MODELS_DIR, MLFLOW_EXPERIMENT, RANDOM_STATE
from src.evaluate import evaluate_model
from src.features import split_features_target


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train model, evaluate it, log to MLflow, and save artifact."""
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": RANDOM_STATE,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics
