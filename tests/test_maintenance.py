import json

import joblib
import pandas as pd
import pytest

import maintenance.activate as activation

from maintenance.prepare_data import validate_dataset
from maintenance.promotion import should_promote


def create_valid_dataset():
    """
    Dataset mínimo válido para probar las funciones
    del pipeline sin depender del dataset completo.
    """
    data = {}

    # El pipeline espera 34 variables predictoras.
    for i in range(34):
        data[f"feature_{i}"] = [
            i,
            i + 1,
            i + 2,
            i + 3,
            i + 4,
            i + 5,
        ]

    data["Target"] = [
        "Graduate",
        "Dropout",
        "Enrolled",
        "Graduate",
        "Dropout",
        "Enrolled",
    ]

    return pd.DataFrame(data)


def test_valid_dataset():
    """
    Caso 1:
    Un dataset válido debe superar la validación.
    """
    df = create_valid_dataset()

    assert validate_dataset(df) is True


def test_dataset_without_target_is_rejected():
    """
    Caso 2:
    El pipeline debe rechazar datos sin variable Target.
    """
    df = create_valid_dataset()
    df = df.drop(columns=["Target"])

    with pytest.raises(
        ValueError,
        match="variable objetivo",
    ):
        validate_dataset(df)


def test_worse_candidate_is_not_promoted():
    """
    Caso 3:
    Un candidato que degrada F1 no debe ser promovido.
    """
    baseline_metrics = {
        "f1": 0.8133,
        "recall": 0.7746,
    }

    candidate_metrics = {
        "f1": 0.8043,
        "recall": 0.7817,
    }

    result = should_promote(
        baseline_metrics,
        candidate_metrics,
    )

    assert result is False


def test_rejected_model_cannot_be_activated():
    """
    Un modelo rechazado por la política de promoción
    no puede pasar a producción.
    """
    with pytest.raises(
        ValueError,
        match="no está aprobada para despliegue",
    ):
        activation.validate_candidate("v1.1.0")


def test_activation_rollback_when_candidate_is_invalid(
    tmp_path,
    monkeypatch,
):
    """
    Comprueba que si un candidato aprobado contiene
    artefactos inválidos, la activación falla y se
    restaura automáticamente el modelo anterior.
    """

    # -------------------------------------------------
    # 1. Crear entorno temporal para la prueba.
    # -------------------------------------------------
    models_dir = tmp_path / "models"
    registry_dir = models_dir / "registry"
    candidate_dir = registry_dir / "v-test"
    backups_dir = models_dir / "backups"

    candidate_dir.mkdir(parents=True)

    active_model = models_dir / "modelo_desercion.pkl"
    active_columns = models_dir / "columnas.pkl"
    production_file = registry_dir / "production.json"

    # -------------------------------------------------
    # 2. Simular el modelo actualmente en producción.
    # -------------------------------------------------
    active_model.write_bytes(
        b"modelo-produccion-original"
    )

    active_columns.write_bytes(
        b"columnas-produccion-original"
    )

    original_production = {
        "production_version": "v-original",
        "model_file": "models/modelo_desercion.pkl",
        "columns_file": "models/columnas.pkl",
    }

    production_file.write_text(
        json.dumps(original_production),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 3. Crear un candidato marcado como aprobado.
    # -------------------------------------------------
    metadata = {
        "version": "v-test",
        "status": "approved",
    }

    (candidate_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Creamos un objeto que joblib puede cargar,
    # pero que NO es un modelo de Machine Learning.
    # Por lo tanto, no implementa predict().
    invalid_candidate = {
        "tipo": "artefacto_invalido",
    }

    joblib.dump(
        invalid_candidate,
        candidate_dir / "modelo.pkl",
    )

    joblib.dump(
        ["feature_1", "feature_2"],
        candidate_dir / "columnas.pkl",
    )

    # -------------------------------------------------
    # 4. Hacer que activate.py use los archivos
    #    temporales y NO los archivos reales.
    # -------------------------------------------------
    monkeypatch.setattr(
        activation,
        "REGISTRY_DIR",
        registry_dir,
    )

    monkeypatch.setattr(
        activation,
        "PRODUCTION_FILE",
        production_file,
    )

    monkeypatch.setattr(
        activation,
        "ACTIVE_MODEL",
        active_model,
    )

    monkeypatch.setattr(
        activation,
        "ACTIVE_COLUMNS",
        active_columns,
    )

    monkeypatch.setattr(
        activation,
        "BACKUP_DIR",
        backups_dir,
    )

    # -------------------------------------------------
    # 5. Intentar activar el candidato inválido.
    #    Debe producir un error.
    # -------------------------------------------------
    with pytest.raises(
        ValueError,
        match="predict",
    ):
        activation.activate("v-test")

    # -------------------------------------------------
    # 6. Verificar que ocurrió el rollback.
    # -------------------------------------------------
    assert (
        active_model.read_bytes()
        == b"modelo-produccion-original"
    )

    assert (
        active_columns.read_bytes()
        == b"columnas-produccion-original"
    )

    restored_production = json.loads(
        production_file.read_text(
            encoding="utf-8",
        )
    )

    assert restored_production == original_production

    # -------------------------------------------------
    # 7. Verificar que se creó el backup.
    # -------------------------------------------------
    backups = list(backups_dir.iterdir())

    assert len(backups) == 1

    assert (
        backups[0] / "modelo_desercion.pkl"
    ).exists()

    assert (
        backups[0] / "columnas.pkl"
    ).exists()

    assert (
        backups[0] / "production.json"
    ).exists()
