"""
Esquemas de datos (Pydantic) para la API de predicción de deserción estudiantil.

Los nombres de los campos del modelo (alias) coinciden exactamente con las
columnas usadas al entrenar el modelo (ver notebooks/entrenamiento.ipynb),
pero los nombres de atributo en Python son válidos (sin espacios ni apóstrofes).
"""

from pydantic import BaseModel, Field, ConfigDict


class StudentFeatures(BaseModel):
    """Datos de un estudiante, usados como entrada para la predicción."""

    model_config = ConfigDict(populate_by_name=True)

    marital_status: int = Field(..., alias="Marital status", ge=1, le=6,
                                 description="Estado civil (código 1-6)")
    application_mode: int = Field(..., alias="Application mode",
                                   description="Modalidad de postulación (código)")
    application_order: int = Field(..., alias="Application order", ge=0, le=9,
                                    description="Orden de preferencia de la postulación")
    course: int = Field(..., alias="Course", description="Código de la carrera")
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance",
                                             description="1=Diurno, 0=Vespertino")
    previous_qualification: int = Field(..., alias="Previous qualification",
                                         description="Código de calificación previa")
    nacionality: int = Field(..., alias="Nacionality", description="Código de nacionalidad")
    mothers_qualification: int = Field(..., alias="Mother's qualification",
                                        description="Código de calificación de la madre")
    fathers_qualification: int = Field(..., alias="Father's qualification",
                                        description="Código de calificación del padre")
    mothers_occupation: int = Field(..., alias="Mother's occupation",
                                     description="Código de ocupación de la madre")
    fathers_occupation: int = Field(..., alias="Father's occupation",
                                     description="Código de ocupación del padre")
    displaced: int = Field(..., alias="Displaced", description="1=Sí, 0=No")
    educational_special_needs: int = Field(..., alias="Educational special needs",
                                            description="1=Sí, 0=No")
    debtor: int = Field(..., alias="Debtor", description="1=Sí, 0=No")
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date",
                                          description="1=Al día, 0=No al día")
    gender: int = Field(..., alias="Gender", description="1=Hombre, 0=Mujer")
    scholarship_holder: int = Field(..., alias="Scholarship holder", description="1=Sí, 0=No")
    age_at_enrollment: int = Field(..., alias="Age at enrollment", ge=15, le=90,
                                    description="Edad al momento de matricularse")
    international: int = Field(..., alias="International", description="1=Sí, 0=No")

    curricular_units_1st_sem_credited: int = Field(..., alias="Curricular units 1st sem (credited)")
    curricular_units_1st_sem_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)")
    curricular_units_1st_sem_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)")
    curricular_units_1st_sem_approved: int = Field(..., alias="Curricular units 1st sem (approved)")
    curricular_units_1st_sem_grade: float = Field(..., alias="Curricular units 1st sem (grade)")
    curricular_units_1st_sem_without_evaluations: int = Field(
        ..., alias="Curricular units 1st sem (without evaluations)")

    curricular_units_2nd_sem_credited: int = Field(..., alias="Curricular units 2nd sem (credited)")
    curricular_units_2nd_sem_enrolled: int = Field(..., alias="Curricular units 2nd sem (enrolled)")
    curricular_units_2nd_sem_evaluations: int = Field(..., alias="Curricular units 2nd sem (evaluations)")
    curricular_units_2nd_sem_approved: int = Field(..., alias="Curricular units 2nd sem (approved)")
    curricular_units_2nd_sem_grade: float = Field(..., alias="Curricular units 2nd sem (grade)")
    curricular_units_2nd_sem_without_evaluations: int = Field(
        ..., alias="Curricular units 2nd sem (without evaluations)")

    unemployment_rate: float = Field(..., alias="Unemployment rate")
    inflation_rate: float = Field(..., alias="Inflation rate")
    gdp: float = Field(..., alias="GDP")


class PredictionResponse(BaseModel):
    """Respuesta de la API para una predicción."""

    prediction: str = Field(..., description='"Dropout" o "No Dropout"')
    dropout_probability: float = Field(..., description="Probabilidad de abandono (0 a 1)")
    risk_level: str = Field(..., description='"Bajo", "Medio" o "Alto"')
    top_factors: list[dict] = Field(
        default_factory=list,
        description="Factores (características) que más influyeron en esta predicción, según SHAP"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
