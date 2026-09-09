"""
Pruebas de funcionamiento de la API de predicción de deserción estudiantil.

Ejecutar con:
    pytest tests/ -v

Estos tests cubren el requisito de "pruebas de funcionamiento" del avance
del proyecto: verifican que la API responde, que el modelo predice de forma
coherente sobre casos conocidos, y que valida correctamente los datos de entrada.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# --- Casos de ejemplo tomados directamente del dataset real ---

ESTUDIANTE_ALTO_RIESGO = {
    "Marital status": 1, "Application mode": 12, "Application order": 1, "Course": 1,
    "Daytime/evening attendance": 1, "Previous qualification": 1, "Nacionality": 1,
    "Mother's qualification": 23, "Father's qualification": 27,
    "Mother's occupation": 10, "Father's occupation": 7,
    "Displaced": 0, "Educational special needs": 0, "Debtor": 1,
    "Tuition fees up to date": 0, "Gender": 1, "Scholarship holder": 0,
    "Age at enrollment": 37, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 7,
    "Curricular units 1st sem (evaluations)": 7, "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0, "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 7,
    "Curricular units 2nd sem (evaluations)": 7, "Curricular units 2nd sem (approved)": 1,
    "Curricular units 2nd sem (grade)": 10.0, "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 8.9, "Inflation rate": 1.4, "GDP": 3.51,
}

ESTUDIANTE_BAJO_RIESGO = {
    "Marital status": 1, "Application mode": 8, "Application order": 2, "Course": 15,
    "Daytime/evening attendance": 1, "Previous qualification": 1, "Nacionality": 1,
    "Mother's qualification": 23, "Father's qualification": 27,
    "Mother's occupation": 6, "Father's occupation": 4,
    "Displaced": 1, "Educational special needs": 0, "Debtor": 0,
    "Tuition fees up to date": 1, "Gender": 0, "Scholarship holder": 0,
    "Age at enrollment": 20, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8, "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 13.43, "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 10, "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 12.4, "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 9.4, "Inflation rate": -0.8, "GDP": -3.12,
}


def test_health_check():
    """La API debe estar viva y con el modelo cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_alto_riesgo():
    """
    Un estudiante con 0 unidades aprobadas el primer semestre, deudor y con
    matrícula atrasada debería predecirse con probabilidad de abandono alta.
    """
    response = client.post("/predict", json=ESTUDIANTE_ALTO_RIESGO)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("Dropout", "No Dropout")
    assert 0 <= data["dropout_probability"] <= 1
    assert data["risk_level"] in ("Bajo", "Medio", "Alto")
    # Se espera que el riesgo no sea "Bajo" dado el perfil claramente adverso
    assert data["risk_level"] in ("Medio", "Alto")
    assert len(data["top_factors"]) == 3


def test_predict_bajo_riesgo():
    """
    Un estudiante con buen desempeño académico, sin deuda y con matrícula
    al día debería predecirse con probabilidad de abandono baja.
    """
    response = client.post("/predict", json=ESTUDIANTE_BAJO_RIESGO)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ("Bajo", "Medio")


def test_predict_datos_invalidos():
    """La API debe rechazar solicitudes con campos faltantes (validación de Pydantic)."""
    datos_incompletos = {"Marital status": 1}  # faltan casi todos los campos requeridos
    response = client.post("/predict", json=datos_incompletos)
    assert response.status_code == 422  # Unprocessable Entity


def test_root_endpoint():
    """El endpoint raíz debe responder con información básica de la API."""
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()
