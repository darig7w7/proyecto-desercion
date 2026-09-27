import pandas as pd
import pytest

from maintenance.prepare_data import (
    validate_dataset,
    create_binary_target,
    split_maintenance_data,
)

from maintenance.promotion import should_promote


def create_valid_dataset():
    """
    Dataset mínimo válido para probar las funciones
    del pipeline sin depender del dataset completo.
    """
    data = {}

    # El pipeline espera 34 variables predictoras.
    for i in range(34):
        data[f"feature_{i}"] = [i, i + 1, i + 2, i + 3, i + 4, i + 5]

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

    with pytest.raises(ValueError, match="variable objetivo"):
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
