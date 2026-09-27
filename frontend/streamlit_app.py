"""
Frontend interactivo para la API de Predicción de Deserción Estudiantil.
Interfaz completamente en español para la predicción y explicación del riesgo.
Los códigos técnicos del dataset se conservan internamente para mantener
compatibilidad con el modelo entrenado.

Ejecutar localmente:
    streamlit run streamlit_app.py
"""

import os

import plotly.graph_objects as go
import requests
import streamlit as st

DEFAULT_API_URL = "https://web-production-ed3cf4.up.railway.app"
API_URL = os.environ.get("API_URL", DEFAULT_API_URL)

st.set_page_config(
    page_title="Predicción del Riesgo de Deserción Estudiantil",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.05rem; font-weight: 600; color: #1F2937; margin-bottom: 0.2rem; }
    .subtitle { color: #6b7280; margin-bottom: 1.5rem; font-size: 0.98rem; }
    .risk-badge {
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 6px;
        font-weight: 600; font-size: 1rem; border: 1px solid transparent;
    }
    .risk-bajo  { background-color: #EAF3EE; color: #2F6D4F; border-color: #CFE3D6; }
    .risk-medio { background-color: #FBF2E3; color: #8A5A1F; border-color: #EFDDBB; }
    .risk-alto  { background-color: #FBEAEA; color: #A13B3B; border-color: #F0C9C9; }
    div.stFormSubmitButton > button {
        background-color: #1F2937; color: #FFFFFF; border: none; font-weight: 500;
    }
    div.stFormSubmitButton > button:hover { background-color: #374151; color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Textos de la interfaz ---
TEXTOS = {
    "title": "Predicción del Riesgo de Deserción Estudiantil",
    "subtitle": "Aprendizaje de Máquina I — UNA Puno · Random Forest + explicabilidad SHAP",
    "load_low": "Cargar ejemplo de bajo riesgo",
    "load_high": "Cargar ejemplo de alto riesgo",
    "own_data": "O ingresa los datos del estudiante",
    "section_personal": "Datos personales y socioeconómicos",
    "section_financial": "Situación financiera y familiar",
    "section_enrollment": "Modalidad de matrícula",
    "section_sem1": "Desempeño académico — 1.er semestre",
    "section_sem2": "Desempeño académico — 2.º semestre",
    "section_macro": "Contexto macroeconómico",
    "marital_status": "Estado civil",
    "application_mode": "Modalidad de postulación",
    "application_order": "Orden de postulación [0-9]",
    "course": "Carrera",
    "gender": "Género",
    "female": "Mujer",
    "male": "Hombre",
    "age": "Edad al matricularse",
    "nationality": "Nacionalidad",
    "displaced": "Desplazado",
    "international": "Internacional",
    "special_needs": "Necesidades educativas especiales",
    "yes": "Sí",
    "no": "No",
    "debtor": "Deudor",
    "tuition_ok": "Matrícula al día",
    "scholarship": "Becario",
    "previous_qual": "Calificación previa",
    "mothers_qual": "Calificación de la madre",
    "fathers_qual": "Calificación del padre",
    "mothers_occ": "Ocupación de la madre",
    "fathers_occ": "Ocupación del padre",
    "attendance": "Turno",
    "evening": "Vespertino",
    "daytime": "Diurno",
    "credited": "Convalidadas",
    "enrolled": "Matriculadas",
    "evaluations": "Evaluaciones",
    "approved": "Aprobadas",
    "grade": "Nota",
    "without_eval": "Sin evaluación",
    "unemployment": "Tasa de desempleo (%)",
    "inflation": "Tasa de inflación (%)",
    "gdp": "PBI",
    "predict_button": "Predecir riesgo de deserción",
    "querying": "Consultando el modelo...",
    "api_error": "No se pudo contactar la API en",
    "result_title": "Resultado de la predicción",
    "prediction": "Predicción",
    "risk_level": "Nivel de riesgo",
    "dropout_prob": "Probabilidad de deserción",
    "gauge_title": "Probabilidad de deserción",
    "shap_title": "Los 3 factores que más influyeron — SHAP",
    "shap_axis": "Impacto SHAP (+ aumenta el riesgo, − disminuye el riesgo)",
    "raw_json": "Respuesta de la API — JSON",
    "connected": "Conectado a la API en:",
    "low": "Bajo",
    "medium": "Medio",
    "high": "Alto",
}

def t(key):
    return TEXTOS[key]

st.markdown(f'<div class="main-title">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)

# --- Diccionarios de categorías en español (códigos originales del dataset) ---


MARITAL_STATUS = {1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a", 4: "Divorciado/a", 5: "Unión de hecho", 6: "Separado/a legalmente"}

NATIONALITY = {
        1: "Portugués", 2: "Alemán", 3: "Español", 4: "Italiano", 5: "Neerlandés", 6: "Inglés",
        7: "Lituano", 8: "Angoleño", 9: "Caboverdiano", 10: "Guineano", 11: "Mozambiqueño",
        12: "Santotomense", 13: "Turco", 14: "Brasileño", 15: "Rumano", 16: "Moldavo",
        17: "Mexicano", 18: "Ucraniano", 19: "Ruso", 20: "Cubano", 21: "Colombiano",
    }

APPLICATION_MODE = {
        1: "1ª fase - contingente general", 2: "Decreto No. 612/93",
        3: "1ª fase - contingente especial (Azores)", 4: "Titulares de otros cursos superiores",
        5: "Decreto No. 854-B/99", 6: "Estudiante internacional (pregrado)",
        7: "1ª fase - contingente especial (Madeira)", 8: "2ª fase - contingente general",
        9: "3ª fase - contingente general", 10: "Decreto 533-A/99, ítem b2 (Plan diferente)",
        11: "Decreto 533-A/99, ítem b3 (Otra institución)", 12: "Mayores de 23 años",
        13: "Traslado", 14: "Cambio de carrera", 15: "Titulares de diploma de especialización tecnológica",
        16: "Cambio de institución/carrera", 17: "Titulares de diploma de ciclo corto",
        18: "Cambio de institución/carrera (Internacional)",
    }

COURSE = {
        1: "Tecnologías de Producción de Biocombustibles", 2: "Diseño de Animación y Multimedia",
        3: "Servicio Social (nocturno)", 4: "Agronomía", 5: "Diseño de Comunicación",
        6: "Enfermería Veterinaria", 7: "Ingeniería Informática", 8: "Equinicultura",
        9: "Gestión", 10: "Servicio Social", 11: "Turismo", 12: "Enfermería",
        13: "Higiene Oral", 14: "Publicidad y Gestión de Marketing",
        15: "Periodismo y Comunicación", 16: "Educación Básica", 17: "Gestión (nocturno)",
    }

PREVIOUS_QUALIFICATION = {
        1: "Educación secundaria", 2: "Educación superior - licenciatura",
        3: "Educación superior - grado", 4: "Educación superior - maestría",
        5: "Educación superior - doctorado", 6: "Frecuencia de educación superior",
        7: "12° año - no completado", 8: "11° año - no completado",
        9: "Otro - 11° año de escolaridad", 10: "10° año de escolaridad",
        11: "10° año - no completado", 12: "Educación básica 3er ciclo (9°-11°)",
        13: "Educación básica 2do ciclo (6°-8°)", 14: "Curso de especialización tecnológica",
        15: "Educación superior - grado (1er ciclo)", 16: "Curso técnico superior profesional",
        17: "Educación superior - maestría (2do ciclo)",
    }

PARENT_QUALIFICATION = {
        1: "Educación secundaria (12°) o equiv.", 2: "Ed. superior - licenciatura",
        3: "Ed. superior - grado", 4: "Ed. superior - maestría", 5: "Ed. superior - doctorado",
        6: "Frecuencia de educación superior", 7: "12° año - no completado",
        8: "11° año - no completado", 9: "7° año (antiguo)", 10: "Otro - 11° año de escolaridad",
        11: "2° año curso complementario secundaria", 12: "10° año de escolaridad",
        13: "Curso general de comercio", 14: "Educación básica 3er ciclo (9°-11°)",
        15: "Curso complementario de secundaria", 16: "Curso técnico-profesional",
        17: "Curso complementario secundaria - no concluido", 18: "7° año de escolaridad",
        19: "2° ciclo curso general secundaria", 20: "9° año - no completado",
        21: "8° año de escolaridad", 22: "Curso general de administración y comercio",
        23: "Contabilidad y administración suplementaria", 24: "Desconocido",
        25: "No sabe leer ni escribir", 26: "Sabe leer sin 4° año completo",
        27: "Educación básica 1er ciclo (4°-5°)", 28: "Educación básica 2do ciclo (6°-8°)",
        29: "Curso de especialización tecnológica", 30: "Educación superior - grado (1er ciclo)",
        31: "Curso de estudios superiores especializados", 32: "Curso técnico superior profesional",
        33: "Ed. superior - maestría (2do ciclo)", 34: "Ed. superior - doctorado (3er ciclo)",
    }

OCCUPATION = {
        1: "Estudiante", 2: "Representantes legislativo/ejecutivo, directores",
        3: "Especialistas en actividades intelectuales/científicas", 4: "Técnicos de nivel intermedio",
        5: "Personal administrativo", 6: "Servicios personales, seguridad, vendedores",
        7: "Agricultores y trabajadores calificados agro/pesca", 8: "Trabajadores calificados industria/construcción",
        9: "Operadores de máquinas y montaje", 10: "Trabajadores no calificados",
        11: "Profesiones de las fuerzas armadas", 12: "Otra situación", 13: "(en blanco)",
        14: "Oficiales de las fuerzas armadas", 15: "Sargentos de las fuerzas armadas",
        16: "Otro personal de fuerzas armadas", 17: "Directores de servicios administrativos/comerciales",
        18: "Directores de hotelería, restauración, comercio", 19: "Especialistas en ciencias físicas/ingeniería",
        20: "Profesionales de la salud", 21: "Docentes", 22: "Especialistas en finanzas/contabilidad",
        23: "Técnicos de ciencias e ingeniería", 24: "Técnicos de nivel intermedio de salud",
        25: "Técnicos legales/sociales/deportivos/culturales", 26: "Técnicos en TIC",
        27: "Oficinistas, secretarios", 28: "Operadores de datos/contabilidad/finanzas",
        29: "Otro personal de apoyo administrativo", 30: "Trabajadores de servicios personales", 31: "Vendedores",
        32: "Trabajadores de cuidado personal", 33: "Personal de protección y seguridad",
        34: "Agricultores orientados al mercado", 35: "Agricultores/pescadores de subsistencia",
        36: "Trabajadores calificados de construcción", 37: "Trabajadores calificados de metalurgia",
        38: "Trabajadores calificados de electricidad/electrónica", 39: "Trabajadores de alimentos/madera/vestido",
        40: "Operadores de planta y máquinas fijas", 41: "Trabajadores de montaje",
        42: "Conductores de vehículos y equipos móviles", 43: "Trabajadores no calificados agro/pesca",
        44: "Trabajadores no calificados extractiva/construcción/transporte", 45: "Auxiliares de preparación de alimentos",
        46: "Vendedores ambulantes (no alimentos) y servicios callejeros",
    }


def selectbox_code(label, category_dict, current_value, key=None):
    options_dict = category_dict
    keys = list(options_dict.keys())
    idx = keys.index(current_value) if current_value in keys else 0
    return st.selectbox(
        label, keys, index=idx,
        format_func=lambda k: f"{k} - {options_dict[k]}",
        key=key,
    )


EJEMPLO_ALTO_RIESGO = {
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
EJEMPLO_BAJO_RIESGO = {
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

if "form_data" not in st.session_state:
    st.session_state.form_data = EJEMPLO_BAJO_RIESGO.copy()

col_ej1, col_ej2, col_ej3 = st.columns([1, 1, 3])
with col_ej1:
    if st.button(t("load_low")):
        st.session_state.form_data = EJEMPLO_BAJO_RIESGO.copy()
with col_ej2:
    if st.button(t("load_high")):
        st.session_state.form_data = EJEMPLO_ALTO_RIESGO.copy()

st.caption(t("own_data"))

d = st.session_state.form_data

with st.form("prediction_form"):

    with st.expander(t("section_personal"), expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            marital_status = selectbox_code(t("marital_status"), MARITAL_STATUS, d["Marital status"])
            application_mode = selectbox_code(t("application_mode"), APPLICATION_MODE, d["Application mode"])
            application_order = st.number_input(t("application_order"), 0, 9, d["Application order"])
        with c2:
            course = selectbox_code(t("course"), COURSE, d["Course"])
            gender = st.selectbox(
                t("gender"), [0, 1], index=d["Gender"],
                format_func=lambda x: t("female") if x == 0 else t("male"),
            )
            age = st.number_input(t("age"), 15, 90, d["Age at enrollment"])
        with c3:
            nacionality = selectbox_code(t("nationality"), NATIONALITY, d["Nacionality"])
            displaced = st.selectbox(
                t("displaced"), [0, 1], index=d["Displaced"],
                format_func=lambda x: t("no") if x == 0 else t("yes"),
            )
            international = st.selectbox(
                t("international"), [0, 1], index=d["International"],
                format_func=lambda x: t("no") if x == 0 else t("yes"),
            )

        special_needs = st.selectbox(
            t("special_needs"), [0, 1], index=d["Educational special needs"],
            format_func=lambda x: t("no") if x == 0 else t("yes"),
        )

    with st.expander(t("section_financial")):
        c1, c2, c3 = st.columns(3)
        with c1:
            debtor = st.selectbox(
                t("debtor"), [0, 1], index=d["Debtor"],
                format_func=lambda x: t("no") if x == 0 else t("yes"),
            )
            tuition_ok = st.selectbox(
                t("tuition_ok"), [0, 1], index=d["Tuition fees up to date"],
                format_func=lambda x: t("no") if x == 0 else t("yes"),
            )
            scholarship = st.selectbox(
                t("scholarship"), [0, 1], index=d["Scholarship holder"],
                format_func=lambda x: t("no") if x == 0 else t("yes"),
            )
            previous_qual = selectbox_code(t("previous_qual"), PREVIOUS_QUALIFICATION, d["Previous qualification"])
        with c2:
            mothers_qual = selectbox_code(t("mothers_qual"), PARENT_QUALIFICATION, d["Mother's qualification"])
            fathers_qual = selectbox_code(t("fathers_qual"), PARENT_QUALIFICATION, d["Father's qualification"])
        with c3:
            mothers_occ = selectbox_code(t("mothers_occ"), OCCUPATION, d["Mother's occupation"])
            fathers_occ = selectbox_code(t("fathers_occ"), OCCUPATION, d["Father's occupation"])

    with st.expander(t("section_enrollment")):
        attendance = st.selectbox(
            t("attendance"), [0, 1], index=d["Daytime/evening attendance"],
            format_func=lambda x: t("evening") if x == 0 else t("daytime"),
        )

    with st.expander(t("section_sem1")):
        c1, c2, c3 = st.columns(3)
        with c1:
            cu1_credited = st.number_input(f'{t("credited")} — 1.er semestre', 0, 30, d["Curricular units 1st sem (credited)"])
            cu1_enrolled = st.number_input(f'{t("enrolled")} — 1.er semestre', 0, 30, d["Curricular units 1st sem (enrolled)"])
        with c2:
            cu1_evaluations = st.number_input(f'{t("evaluations")} — 1.er semestre', 0, 30, d["Curricular units 1st sem (evaluations)"])
            cu1_approved = st.number_input(f'{t("approved")} — 1.er semestre', 0, 30, d["Curricular units 1st sem (approved)"])
        with c3:
            cu1_grade = st.number_input(f'{t("grade")} — 1.er semestre', 0.0, 20.0, float(d["Curricular units 1st sem (grade)"]))
            cu1_without_eval = st.number_input(f'{t("without_eval")} — 1.er semestre', 0, 30, d["Curricular units 1st sem (without evaluations)"])

    with st.expander(t("section_sem2")):
        c1, c2, c3 = st.columns(3)
        with c1:
            cu2_credited = st.number_input(f'{t("credited")} — 2.º semestre', 0, 30, d["Curricular units 2nd sem (credited)"])
            cu2_enrolled = st.number_input(f'{t("enrolled")} — 2.º semestre', 0, 30, d["Curricular units 2nd sem (enrolled)"])
        with c2:
            cu2_evaluations = st.number_input(f'{t("evaluations")} — 2.º semestre', 0, 30, d["Curricular units 2nd sem (evaluations)"])
            cu2_approved = st.number_input(f'{t("approved")} — 2.º semestre', 0, 30, d["Curricular units 2nd sem (approved)"])
        with c3:
            cu2_grade = st.number_input(f'{t("grade")} — 2.º semestre', 0.0, 20.0, float(d["Curricular units 2nd sem (grade)"]))
            cu2_without_eval = st.number_input(f'{t("without_eval")} — 2.º semestre', 0, 30, d["Curricular units 2nd sem (without evaluations)"])

    with st.expander(t("section_macro")):
        c1, c2, c3 = st.columns(3)
        with c1:
            unemployment = st.number_input(t("unemployment"), -20.0, 30.0, float(d["Unemployment rate"]))
        with c2:
            inflation = st.number_input(t("inflation"), -20.0, 30.0, float(d["Inflation rate"]))
        with c3:
            gdp = st.number_input(t("gdp"), -20.0, 20.0, float(d["GDP"]))

    submitted = st.form_submit_button(t("predict_button"), use_container_width=True)

if submitted:
    payload = {
        "Marital status": marital_status, "Application mode": application_mode,
        "Application order": application_order, "Course": course,
        "Daytime/evening attendance": attendance, "Previous qualification": previous_qual,
        "Nacionality": nacionality, "Mother's qualification": mothers_qual,
        "Father's qualification": fathers_qual, "Mother's occupation": mothers_occ,
        "Father's occupation": fathers_occ, "Displaced": displaced,
        "Educational special needs": special_needs, "Debtor": debtor,
        "Tuition fees up to date": tuition_ok, "Gender": gender,
        "Scholarship holder": scholarship, "Age at enrollment": age,
        "International": international,
        "Curricular units 1st sem (credited)": cu1_credited,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (evaluations)": cu1_evaluations,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_without_eval,
        "Curricular units 2nd sem (credited)": cu2_credited,
        "Curricular units 2nd sem (enrolled)": cu2_enrolled,
        "Curricular units 2nd sem (evaluations)": cu2_evaluations,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_grade,
        "Curricular units 2nd sem (without evaluations)": cu2_without_eval,
        "Unemployment rate": unemployment, "Inflation rate": inflation, "GDP": gdp,
    }

    try:
        with st.spinner(t("querying")):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"{t('api_error')} {API_URL}. Detalle: {e}")
        st.stop()

    st.divider()
    st.subheader(t("result_title"))

    risk_level = result["risk_level"]
    risk_label = {"Bajo": t("low"), "Medio": t("medium"), "Alto": t("high")}[risk_level]
    risk_class = {"Bajo": "risk-bajo", "Medio": "risk-medio", "Alto": "risk-alto"}[risk_level]
    risk_color = {"Bajo": "#16a34a", "Medio": "#ca8a04", "Alto": "#dc2626"}[risk_level]

    result_box = st.container(border=True)
    with result_box:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            prediction_es = {
                "Dropout": "Deserción",
                "No Dropout": "No deserción",
            }.get(result["prediction"], result["prediction"])
            st.markdown(f"**{t('prediction')}:** {prediction_es}")
            st.markdown(
                f'**{t("risk_level")}:** <span class="risk-badge {risk_class}">{risk_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{t('dropout_prob')}:** {result['dropout_probability'] * 100:.1f}%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["dropout_probability"] * 100,
                number={"suffix": "%"},
                title={"text": t("gauge_title")},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 33], "color": "#dcfce7"},
                        {"range": [33, 66], "color": "#fef9c3"},
                        {"range": [66, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge, use_container_width=True)

        with col_right:
            st.markdown(f"**{t('shap_title')}**")
            factors = result["top_factors"]
            FEATURE_NAMES_ES = {
    "Curricular units 1st sem (credited)": "Unidades convalidadas — 1.er semestre",
    "Curricular units 1st sem (enrolled)": "Unidades matriculadas — 1.er semestre",
    "Curricular units 1st sem (evaluations)": "Evaluaciones — 1.er semestre",
    "Curricular units 1st sem (approved)": "Unidades aprobadas — 1.er semestre",
    "Curricular units 1st sem (grade)": "Nota — 1.er semestre",
    "Curricular units 1st sem (without evaluations)": "Unidades sin evaluación — 1.er semestre",
    "Curricular units 2nd sem (credited)": "Unidades convalidadas — 2.º semestre",
    "Curricular units 2nd sem (enrolled)": "Unidades matriculadas — 2.º semestre",
    "Curricular units 2nd sem (evaluations)": "Evaluaciones — 2.º semestre",
    "Curricular units 2nd sem (approved)": "Unidades aprobadas — 2.º semestre",
    "Curricular units 2nd sem (grade)": "Nota — 2.º semestre",
    "Curricular units 2nd sem (without evaluations)": "Unidades sin evaluación — 2.º semestre",
    "Age at enrollment": "Edad al matricularse",
    "Tuition fees up to date": "Matrícula al día",
    "Debtor": "Deudor",
    "Scholarship holder": "Becario",
    "Gender": "Género",
    "Application mode": "Modalidad de postulación",
    "Application order": "Orden de postulación",
    "Course": "Carrera",
    "Previous qualification": "Calificación previa",
    "Unemployment rate": "Tasa de desempleo",
    "Inflation rate": "Tasa de inflación",
    "GDP": "PBI",
}
            names = [
                FEATURE_NAMES_ES.get(f["feature"], f["feature"])
                for f in factors
            ][::-1]
            impacts = [f["impacto"] for f in factors][::-1]
            colors = ["#dc2626" if v > 0 else "#16a34a" for v in impacts]

            bar = go.Figure(go.Bar(x=impacts, y=names, orientation="h", marker_color=colors))
            bar.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title=t("shap_axis"),
            )
            st.plotly_chart(bar, use_container_width=True)

    with st.expander(t("raw_json")):
        st.json(result)

st.divider()
st.caption(f"{t('connected')} {API_URL}")