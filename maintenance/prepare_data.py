from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("notebooks/dataset.csv")
OUTPUT_DIR = Path("maintenance/data")

TARGET_COLUMN = "Target"
RANDOM_STATE = 42

VALID_TARGETS = {"Graduate", "Dropout", "Enrolled"}


def load_dataset(path=DATASET_PATH):
    """Carga el dataset original."""
    return pd.read_csv(path)


def validate_dataset(df):
    """
    Valida las condiciones mínimas necesarias
    antes de utilizar datos en mantenimiento.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"El dataset no contiene la variable objetivo '{TARGET_COLUMN}'."
        )

    if df.isnull().any().any():
        raise ValueError("El dataset contiene valores nulos.")

    targets = set(df[TARGET_COLUMN].unique())

    if not targets.issubset(VALID_TARGETS):
        raise ValueError(
            f"Se encontraron valores no válidos en Target: {targets}"
        )

    if len(df.columns) != 35:
        raise ValueError(
            f"Se esperaban 35 columnas y se encontraron {len(df.columns)}."
        )

    return True


def create_binary_target(df):
    """
    Convierte el problema original de tres clases
    en clasificación binaria:

    Dropout     -> 1
    Graduate    -> 0
    Enrolled    -> 0
    """
    df = df.copy()
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Dropout").astype(int)

    return df


def split_maintenance_data(df):
    """
    Divide los datos de manera estratificada:

    60% historical
    20% new batch
    20% holdout

    El holdout permanece aislado para comparar
    las diferentes versiones del modelo.
    """

    # Primero reservamos 20% como holdout.
    development, holdout = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    # Del 80% restante obtenemos:
    # 75% = 60% del total (historical)
    # 25% = 20% del total (new batch)
    historical, new_batch = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=development[TARGET_COLUMN],
    )

    return historical, new_batch, holdout


def save_datasets(historical, new_batch, holdout):
    """Guarda las particiones para hacer reproducible el experimento."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical.to_csv(OUTPUT_DIR / "historical.csv", index=False)
    new_batch.to_csv(OUTPUT_DIR / "new_batch.csv", index=False)
    holdout.to_csv(OUTPUT_DIR / "holdout.csv", index=False)


def main():
    print("=== PREPARACIÓN DE DATOS PARA MANTENIMIENTO ===")

    df = load_dataset()

    print(f"Dataset original: {len(df)} registros")

    validate_dataset(df)
    print("Validación: OK")

    df = create_binary_target(df)

    print("\nTarget binario:")
    print(df[TARGET_COLUMN].value_counts())

    historical, new_batch, holdout = split_maintenance_data(df)

    save_datasets(historical, new_batch, holdout)

    print("\n=== PARTICIONES ===")
    print(f"Historical : {len(historical)}")
    print(f"New batch  : {len(new_batch)}")
    print(f"Holdout    : {len(holdout)}")
    print(f"Total      : {len(historical) + len(new_batch) + len(holdout)}")

    print("\nArchivos generados en:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
