# Student Dropout Risk Prediction

**Universidad Nacional del Altiplano - Puno | Ingeniería de Sistemas | Maestría en Ciencia de Datos**
**Course:** Machine Learning I - Group A | **Assignment:** Course Project (Unit I - Progress Delivery)

## What this application does

This application predicts whether a university student is at risk of **dropping out**, based on data available at enrollment time and after the first two semesters (academic performance, socioeconomic background, and demographic information).

Given a student's data, the API returns:
- A prediction: `Dropout` or `No Dropout`
- The probability of dropout (0 to 1)
- A risk level: `Bajo` (Low) / `Medio` (Medium) / `Alto` (High)
- The top 3 factors that most influenced that specific prediction (via SHAP explainability)

## Dataset

[**Predict Students' Dropout and Academic Success**](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) — UCI Machine Learning Repository (Realinho et al., 2021), CC BY 4.0 license.

- 4,424 students, 34 features (demographic, socioeconomic, macroeconomic, and academic performance)
- Original problem: 3 classes (Dropout / Enrolled / Graduate) — simplified here to **binary classification** (Dropout vs. No Dropout) to match the application's purpose: flagging at-risk students.

## Model

- **Algorithm:** Random Forest Classifier, hyperparameters optimized via `GridSearchCV` (5-fold CV, optimizing F1-score)
- **Why Random Forest** over the marginally higher-F1 Logistic Regression: tree-based models support `shap.TreeExplainer`, providing fast and exact per-prediction explanations — a core requirement for this application.
- **Class imbalance handling:** `class_weight="balanced"`, since ~32% of students in the dataset dropped out.
- **Explainability:** SHAP (SHapley Additive exPlanations) — both global feature importance (training notebook) and per-prediction explanations (API response).

Full training process, model comparison, hyperparameter search, and SHAP analysis: see [`notebooks/entrenamiento.ipynb`](notebooks/entrenamiento.ipynb).

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI application (endpoints)
│   ├── schemas.py        # Pydantic request/response schemas
│   └── __init__.py
├── models/
│   ├── modelo_desercion.pkl   # Trained model (Random Forest)
│   └── columnas.pkl           # Expected feature order
├── notebooks/
│   ├── entrenamiento.ipynb    # Full training + evaluation + SHAP notebook
│   └── dataset.csv
├── tests/
│   └── test_api.py       # Functional tests (pytest)
├── requirements.txt
├── railway.json
├── Procfile
└── runtime.txt
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:
- `http://localhost:8000/docs` — interactive Swagger UI to try the API
- `http://localhost:8000/health` — health check

## API Endpoints

| Method | Endpoint    | Description                                  |
|--------|-------------|-----------------------------------------------|
| GET    | `/`         | Basic API info                                |
| GET    | `/health`   | Health check (API + model status)             |
| POST   | `/predict`  | Predict dropout risk for one student          |

### Example request to `/predict`

```json
{
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
  "Unemployment rate": 9.4, "Inflation rate": -0.8, "GDP": -3.12
}
```

### Example response

```json
{
  "prediction": "No Dropout",
  "dropout_probability": 0.2379,
  "risk_level": "Bajo",
  "top_factors": [
    {"feature": "Curricular units 2nd sem (approved)", "impacto": -0.0749},
    {"feature": "Curricular units 1st sem (approved)", "impacto": -0.068},
    {"feature": "Curricular units 2nd sem (grade)", "impacto": -0.0634}
  ]
}
```

## Testing

```bash
pytest tests/ -v
```

Covers: health check, prediction on a known high-risk student profile, prediction on a known low-risk student profile, input validation, and root endpoint.

## Deployment

Deployed on [Railway](https://railway.app), connected to this GitHub repository for automatic redeployment on every push to `main`.

## Roadmap (Unit II - Final Delivery)

- Automated retraining pipeline
- Model registry (versioning + metrics tracking)
- CI pipeline (GitHub Actions): run tests automatically before deploy
- CD pipeline: automatic redeploy to Railway after tests pass

## References

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student Dropout and Academic Success. *Data*, 7(11), 146. https://doi.org/10.3390/data7110146
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
