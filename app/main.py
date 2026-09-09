"""
API de predicción de deserción estudiantil.

Carga el modelo entrenado (Random Forest, ver notebooks/entrenamiento.ipynb)
y expone endpoints para:
  - GET  /health   -> verificar que la API y el modelo están operativos
  - POST /predict  -> predecir riesgo de deserción para un estudiante

Ejecutar localmente:
    uvicorn app.main:app --reload

Documentación interactiva (Swagger UI) disponible en /docs una vez desplegado.
"""

import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import StudentFeatures, PredictionResponse, HealthResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_desercion.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "columnas.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se ejecuta al iniciar la aplicación: precarga el modelo en memoria
    _cargar_modelo()
    yield
    # (nada que limpiar al apagar)


app = FastAPI(
    title="API de Predicción de Deserción Estudiantil",
    description=(
        "Predice el riesgo de deserción de un estudiante universitario a partir de "
        "datos demográficos, socioeconómicos y de desempeño académico. "
        "Proyecto del curso Aprendizaje de Máquina I — UNA Puno."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permite que un frontend (Streamlit, HTML, etc.) en otro dominio consuma la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Carga del modelo al iniciar la aplicación ---
_modelo = None
_columnas = None
_explainer = None


def _cargar_modelo():
    global _modelo, _columnas, _explainer
    if _modelo is None:
        _modelo = joblib.load(MODEL_PATH)
        _columnas = joblib.load(COLUMNS_PATH)
        _explainer = shap.TreeExplainer(_modelo)
    return _modelo, _columnas, _explainer


@app.get("/", tags=["General"])
def root():
    return {
        "mensaje": "API de Predicción de Deserción Estudiantil",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Verifica que la API está viva y el modelo cargado correctamente."""
    try:
        modelo, _, _ = _cargar_modelo()
        return HealthResponse(status="ok", model_loaded=modelo is not None)
    except Exception:
        return HealthResponse(status="error", model_loaded=False)


def _clasificar_riesgo(prob: float) -> str:
    if prob < 0.33:
        return "Bajo"
    elif prob < 0.66:
        return "Medio"
    return "Alto"


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(student: StudentFeatures):
    """
    Recibe los datos de un estudiante y devuelve:
      - la predicción (Dropout / No Dropout)
      - la probabilidad de abandono
      - un nivel de riesgo (Bajo/Medio/Alto)
      - los 3 factores que más influyeron en la predicción (explicabilidad SHAP)
    """
    try:
        modelo, columnas, explainer = _cargar_modelo()

        # Se usa by_alias para obtener las claves con los nombres originales
        # del dataset (con espacios/apóstrofes), tal como los espera el modelo.
        datos = student.model_dump(by_alias=True)
        fila = pd.DataFrame([datos])[columnas]

        pred = modelo.predict(fila)[0]
        proba = modelo.predict_proba(fila)[0][1]

        # Explicabilidad: contribución de cada característica a esta predicción
        shap_values = explainer.shap_values(fila)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif np.ndim(shap_values) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        contribuciones = sorted(
            zip(columnas, sv), key=lambda x: abs(x[1]), reverse=True
        )[:3]
        top_factors = [
            {"feature": nombre, "impacto": round(float(valor), 4)}
            for nombre, valor in contribuciones
        ]

        return PredictionResponse(
            prediction="Dropout" if pred == 1 else "No Dropout",
            dropout_probability=round(float(proba), 4),
            risk_level=_clasificar_riesgo(proba),
            top_factors=top_factors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")
