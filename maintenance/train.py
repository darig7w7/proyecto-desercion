from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_DIR = Path("maintenance/data")
REGISTRY_DIR = Path("models/registry")

TARGET_COLUMN = "Target"
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_partitions():
    """Carga las tres particiones creadas por prepare_data.py."""
    historical = pd.read_csv(DATA_DIR / "historical.csv")
    new_batch = pd.read_csv(DATA_DIR / "new_batch.csv")
    holdout = pd.read_csv(DATA_DIR / "holdout.csv")

    return historical, new_batch, holdout


def split_xy(df):
    """Separa variables predictoras y variable objetivo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def train_model(X, y):
    """Entrena un Random Forest con los parámetros definidos."""
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo sobre un conjunto que no participó en entrenamiento."""
    predictions = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
    }


def save_version(version, status, model, columns, metrics, training_rows):
    """Registra el artefacto y sus metadatos."""

    version_dir = REGISTRY_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "modelo.pkl"
    columns_path = version_dir / "columnas.pkl"
    metadata_path = version_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(columns, columns_path)

    metadata = {
        "version": version,
        "status": status,
        "algorithm": "RandomForestClassifier",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(training_rows),
        "n_features": int(len(columns)),
        "hyperparameters": MODEL_PARAMS,
        "metrics": metrics,
        "evaluation_dataset": "maintenance/data/holdout.csv",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    print("=== ENTRENAMIENTO Y VERSIONAMIENTO ===")

    historical, new_batch, holdout = load_partitions()

    X_historical, y_historical = split_xy(historical)
    X_new, y_new = split_xy(new_batch)
    X_holdout, y_holdout = split_xy(holdout)

    print("\nEntrenando baseline v1.0...")
    baseline = train_model(X_historical, y_historical)

    baseline_metrics = evaluate_model(
        baseline,
        X_holdout,
        y_holdout,
    )

    print("Métricas v1.0:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Simulamos que llegaron nuevos datos etiquetados.
    X_updated = pd.concat(
        [X_historical, X_new],
        ignore_index=True,
    )

    y_updated = pd.concat(
        [y_historical, y_new],
        ignore_index=True,
    )

    print("\nEntrenando candidato v1.1...")
    candidate = train_model(X_updated, y_updated)

    candidate_metrics = evaluate_model(
        candidate,
        X_holdout,
        y_holdout,
    )

    print("Métricas v1.1:")
    for metric, value in candidate_metrics.items():
        print(f"  {metric}: {value:.4f}")

    columns = list(X_historical.columns)

    save_version(
        version="v1.0.0-baseline",
        status="baseline",
        model=baseline,
        columns=columns,
        metrics=baseline_metrics,
        training_rows=len(historical),
    )

    save_version(
        version="v1.1.0",
        status="candidate",
        model=candidate,
        columns=columns,
        metrics=candidate_metrics,
        training_rows=len(X_updated),
    )

    print("\n=== COMPARACIÓN ===")
    print(
        f"F1 baseline : {baseline_metrics['f1']:.4f}"
    )
    print(
        f"F1 candidate: {candidate_metrics['f1']:.4f}"
    )

    difference = candidate_metrics["f1"] - baseline_metrics["f1"]

    print(f"Diferencia F1: {difference:+.4f}")

    print("\nModelos registrados correctamente.")


if __name__ == "__main__":
    main()

