"""
Frontend interactivo para la API de Predicción de Deserción Estudiantil.
Includes a language toggle (English / Español) for all UI labels.
Category value lists (course, occupation, qualification, nationality,
application mode) are kept in English in both modes — they are technical
codes from the original dataset and are shown with their numeric code so
they remain unambiguous regardless of language.

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
    page_title="Student Dropout Risk Predictor",
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

# --- Idioma ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"

top_l, top_r = st.columns([5, 1])
with top_r:
    lang_choice = st.radio(
        "Language / Idioma", ["English", "Español"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state.lang = "en" if lang_choice == "English" else "es"

LANG = st.session_state.lang


def t(key):
    return TEXTS[key][LANG]


TEXTS = {
    "title": {"en": "Student Dropout Risk Predictor", "es": "Predictor de Riesgo de Deserción"},
    "subtitle": {
        "en": "Machine Learning I — UNA Puno · Random Forest + SHAP explainability",
        "es": "Aprendizaje de Máquina I — UNA Puno · Random Forest + explicabilidad SHAP",
    },
    "load_low": {"en": "Load low-risk example", "es": "Cargar ejemplo bajo riesgo"},
    "load_high": {"en": "Load high-risk example", "es": "Cargar ejemplo alto riesgo"},
    "own_data": {
        "en": "Or fill in your own student data below",
        "es": "O ingresa los datos de tu propio estudiante abajo",
    },
    "section_personal": {"en": "Personal & Socioeconomic Data", "es": "Datos personales y socioeconómicos"},
    "section_financial": {"en": "Financial & Family Background", "es": "Situación financiera y familiar"},
    "section_enrollment": {"en": "Enrollment Mode", "es": "Modalidad de matrícula"},
    "section_sem1": {"en": "Academic Performance — 1st Semester", "es": "Desempeño académico — 1er semestre"},
    "section_sem2": {"en": "Academic Performance — 2nd Semester", "es": "Desempeño académico — 2do semestre"},
    "section_macro": {"en": "Macroeconomic Context", "es": "Contexto macroeconómico"},
    "marital_status": {"en": "Marital status", "es": "Estado civil"},
    "application_mode": {"en": "Application mode", "es": "Modalidad de postulación"},
    "application_order": {"en": "Application order [0-9]", "es": "Orden de postulación [0-9]"},
    "course": {"en": "Course", "es": "Carrera"},
    "gender": {"en": "Gender", "es": "Género"},
    "female": {"en": "Female", "es": "Mujer"},
    "male": {"en": "Male", "es": "Hombre"},
    "age": {"en": "Age at enrollment", "es": "Edad al matricularse"},
    "nationality": {"en": "Nacionality", "es": "Nacionalidad"},
    "displaced": {"en": "Displaced", "es": "Desplazado"},
    "international": {"en": "International", "es": "Internacional"},
    "special_needs": {"en": "Educational special needs", "es": "Necesidades educativas especiales"},
    "yes": {"en": "Yes", "es": "Sí"},
    "no": {"en": "No", "es": "No"},
    "debtor": {"en": "Debtor", "es": "Deudor"},
    "tuition_ok": {"en": "Tuition fees up to date", "es": "Matrícula al día"},
    "scholarship": {"en": "Scholarship holder", "es": "Becario"},
    "previous_qual": {"en": "Previous qualification", "es": "Calificación previa"},
    "mothers_qual": {"en": "Mother's qualification", "es": "Calificación de la madre"},
    "fathers_qual": {"en": "Father's qualification", "es": "Calificación del padre"},
    "mothers_occ": {"en": "Mother's occupation", "es": "Ocupación de la madre"},
    "fathers_occ": {"en": "Father's occupation", "es": "Ocupación del padre"},
    "attendance": {"en": "Daytime/evening attendance", "es": "Turno"},
    "evening": {"en": "Evening", "es": "Vespertino"},
    "daytime": {"en": "Daytime", "es": "Diurno"},
    "credited": {"en": "Credited", "es": "Convalidadas"},
    "enrolled": {"en": "Enrolled", "es": "Matriculadas"},
    "evaluations": {"en": "Evaluations", "es": "Evaluaciones"},
    "approved": {"en": "Approved", "es": "Aprobadas"},
    "grade": {"en": "Grade", "es": "Nota"},
    "without_eval": {"en": "Without evaluations", "es": "Sin evaluación"},
    "unemployment": {"en": "Unemployment rate (%)", "es": "Tasa de desempleo (%)"},
    "inflation": {"en": "Inflation rate (%)", "es": "Tasa de inflación (%)"},
    "gdp": {"en": "GDP", "es": "PBI"},
    "predict_button": {"en": "Predict dropout risk", "es": "Predecir riesgo de deserción"},
    "querying": {"en": "Querying the model...", "es": "Consultando el modelo..."},
    "api_error": {"en": "Could not reach the API at", "es": "No se pudo contactar la API en"},
    "result_title": {"en": "Prediction Result", "es": "Resultado de la predicción"},
    "prediction": {"en": "Prediction", "es": "Predicción"},
    "risk_level": {"en": "Risk level", "es": "Nivel de riesgo"},
    "dropout_prob": {"en": "Dropout probability", "es": "Probabilidad de abandono"},
    "gauge_title": {"en": "Dropout probability", "es": "Probabilidad de deserción"},
    "shap_title": {"en": "Top 3 factors behind this prediction — SHAP", "es": "Los 3 factores que más influyeron — SHAP"},
    "shap_axis": {
        "en": "SHAP impact (+ pushes to Dropout, − pushes to No Dropout)",
        "es": "Impacto SHAP (+ empuja a Dropout, − empuja a No Dropout)",
    },
    "raw_json": {"en": "Raw API response — JSON", "es": "Respuesta cruda de la API — JSON"},
    "connected": {"en": "Connected to API at:", "es": "Conectado a la API en:"},
    "low": {"en": "Low", "es": "Bajo"},
    "medium": {"en": "Medium", "es": "Medio"},
    "high": {"en": "High", "es": "Alto"},
}

st.markdown(f'<div class="main-title">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)

# --- Diccionarios de categorías (verificados contra notebooks/dataset.csv) ---
# Se muestran en inglés en ambos idiomas, con el código numérico visible.

MARITAL_STATUS = {
    1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
    5: "Facto union", 6: "Legally separated",
}
NATIONALITY = {
    1: "Portuguese", 2: "German", 3: "Spanish", 4: "Italian", 5: "Dutch",
    6: "English", 7: "Lithuanian", 8: "Angolan", 9: "Cape Verdean", 10: "Guinean",
    11: "Mozambican", 12: "Santomean", 13: "Turkish", 14: "Brazilian", 15: "Romanian",
    16: "Moldovan", 17: "Mexican", 18: "Ukrainian", 19: "Russian", 20: "Cuban",
    21: "Colombian",
}
APPLICATION_MODE = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
    3: "1st phase - special contingent (Azores)", 4: "Holders of other higher courses",
    5: "Ordinance No. 854-B/99", 6: "International student (bachelor)",
    7: "1st phase - special contingent (Madeira)", 8: "2nd phase - general contingent",
    9: "3rd phase - general contingent", 10: "Ordinance 533-A/99, b2 (Different Plan)",
    11: "Ordinance 533-A/99, b3 (Other Institution)", 12: "Over 23 years old",
    13: "Transfer", 14: "Change of course", 15: "Technological specialization diploma holders",
    16: "Change of institution/course", 17: "Short cycle diploma holders",
    18: "Change of institution/course (International)",
}
COURSE = {
    1: "Biofuel Production Technologies", 2: "Animation and Multimedia Design",
    3: "Social Service (evening)", 4: "Agronomy", 5: "Communication Design",
    6: "Veterinary Nursing", 7: "Informatics Engineering", 8: "Equinculture",
    9: "Management", 10: "Social Service", 11: "Tourism", 12: "Nursing",
    13: "Oral Hygiene", 14: "Advertising and Marketing Management",
    15: "Journalism and Communication", 16: "Basic Education",
    17: "Management (evening)",
}
PREVIOUS_QUALIFICATION = {
    1: "Secondary education", 2: "Higher education - bachelor's degree",
    3: "Higher education - degree", 4: "Higher education - master's degree",
    5: "Higher education - doctorate", 6: "Frequency of higher education",
    7: "12th year - not completed", 8: "11th year - not completed",
    9: "Other - 11th year of schooling", 10: "10th year of schooling",
    11: "10th year - not completed", 12: "Basic education 3rd cycle (9th-11th)",
    13: "Basic education 2nd cycle (6th-8th)", 14: "Technological specialization course",
    15: "Higher education - degree (1st cycle)", 16: "Professional higher technical course",
    17: "Higher education - master's (2nd cycle)",
}
PARENT_QUALIFICATION = {
    1: "Secondary Education (12th year) or Eq.", 2: "Higher Ed. - bachelor's degree",
    3: "Higher Ed. - degree", 4: "Higher Ed. - master's degree", 5: "Higher Ed. - doctorate",
    6: "Frequency of Higher Education", 7: "12th year - not completed",
    8: "11th year - not completed", 9: "7th Year (Old)", 10: "Other - 11th year of schooling",
    11: "2nd year complementary high school", 12: "10th year of schooling",
    13: "General commerce course", 14: "Basic education 3rd cycle (9th-11th)",
    15: "Complementary High School Course", 16: "Technical-professional course",
    17: "Complementary HS Course - not concluded", 18: "7th year of schooling",
    19: "2nd cycle general high school course", 20: "9th year - not completed",
    21: "8th year of schooling", 22: "General Course of Admin. and Commerce",
    23: "Supplementary Accounting and Admin.", 24: "Unknown",
    25: "Cannot read or write", 26: "Can read, no 4th year of schooling",
    27: "Basic education 1st cycle (4th-5th)", 28: "Basic education 2nd cycle (6th-8th)",
    29: "Technological specialization course", 30: "Higher education - degree (1st cycle)",
    31: "Specialized higher studies course", 32: "Professional higher technical course",
    33: "Higher Ed. - master's (2nd cycle)", 34: "Higher Ed. - doctorate (3rd cycle)",
}
OCCUPATION = {
    1: "Student", 2: "Legislative/Executive reps, Directors",
    3: "Intellectual & Scientific Specialists", 4: "Intermediate Level Technicians",
    5: "Administrative staff", 6: "Personal Services, Security, Sellers",
    7: "Farmers & Skilled Agri/Fishery Workers", 8: "Skilled Industry/Construction Workers",
    9: "Machine Operators & Assembly Workers", 10: "Unskilled Workers",
    11: "Armed Forces Professions", 12: "Other Situation", 13: "(blank)",
    14: "Armed Forces Officers", 15: "Armed Forces Sergeants",
    16: "Other Armed Forces personnel", 17: "Directors of admin./commercial services",
    18: "Hotel, catering, trade directors", 19: "Physical sciences/engineering specialists",
    20: "Health professionals", 21: "Teachers", 22: "Finance/accounting specialists",
    23: "Science & engineering technicians", 24: "Intermediate health technicians",
    25: "Legal/social/sports/cultural technicians", 26: "ICT technicians",
    27: "Office workers, secretaries", 28: "Data/accounting/financial operators",
    29: "Other administrative support", 30: "Personal service workers", 31: "Sellers",
    32: "Personal care workers", 33: "Protection & security personnel",
    34: "Market-oriented farmers", 35: "Subsistence farmers/fishers/hunters",
    36: "Skilled construction workers", 37: "Skilled metallurgy workers",
    38: "Skilled electricity/electronics workers", 39: "Food/wood/clothing industry workers",
    40: "Fixed plant/machine operators", 41: "Assembly workers",
    42: "Vehicle drivers & equipment operators", 43: "Unskilled agri/fishery workers",
    44: "Unskilled extractive/construction/transport", 45: "Meal preparation assistants",
    46: "Street vendors (non-food) & street services",
}


def selectbox_code(label, options_dict, current_value, key=None):
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
            cu1_credited = st.number_input(f'{t("credited")} — 1st sem', 0, 30, d["Curricular units 1st sem (credited)"])
            cu1_enrolled = st.number_input(f'{t("enrolled")} — 1st sem', 0, 30, d["Curricular units 1st sem (enrolled)"])
        with c2:
            cu1_evaluations = st.number_input(f'{t("evaluations")} — 1st sem', 0, 30, d["Curricular units 1st sem (evaluations)"])
            cu1_approved = st.number_input(f'{t("approved")} — 1st sem', 0, 30, d["Curricular units 1st sem (approved)"])
        with c3:
            cu1_grade = st.number_input(f'{t("grade")} — 1st sem', 0.0, 20.0, float(d["Curricular units 1st sem (grade)"]))
            cu1_without_eval = st.number_input(f'{t("without_eval")} — 1st sem', 0, 30, d["Curricular units 1st sem (without evaluations)"])

    with st.expander(t("section_sem2")):
        c1, c2, c3 = st.columns(3)
        with c1:
            cu2_credited = st.number_input(f'{t("credited")} — 2nd sem', 0, 30, d["Curricular units 2nd sem (credited)"])
            cu2_enrolled = st.number_input(f'{t("enrolled")} — 2nd sem', 0, 30, d["Curricular units 2nd sem (enrolled)"])
        with c2:
            cu2_evaluations = st.number_input(f'{t("evaluations")} — 2nd sem', 0, 30, d["Curricular units 2nd sem (evaluations)"])
            cu2_approved = st.number_input(f'{t("approved")} — 2nd sem', 0, 30, d["Curricular units 2nd sem (approved)"])
        with c3:
            cu2_grade = st.number_input(f'{t("grade")} — 2nd sem', 0.0, 20.0, float(d["Curricular units 2nd sem (grade)"]))
            cu2_without_eval = st.number_input(f'{t("without_eval")} — 2nd sem', 0, 30, d["Curricular units 2nd sem (without evaluations)"])

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
        st.error(f"{t('api_error')} {API_URL}. Detail: {e}")
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
            st.markdown(f"**{t('prediction')}:** {result['prediction']}")
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
            names = [f["feature"] for f in factors][::-1]
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
