import streamlit as st
import numpy as np
import joblib
import os
 
# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="PA2: Fundamentos de Machine Learning",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Mulish:wght@300;400;600&display=swap');
 
*, *::before, *::after { box-sizing: border-box; }
 
html, body, [class*="css"] {
    font-family: 'Mulish', sans-serif;
    background-color: #0d0f14;
    color: #e8e8e8;
}
 
.stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111827 60%, #0d1520 100%);
}
 
/* ── Header ── */
.main-header {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem;
    border: 1px solid rgba(99,202,183,0.2);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "🏠";
    position: absolute;
    right: 2.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5.5rem;
    opacity: 0.10;
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #63cab7;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8fa8b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
 
/* ── Caja estudiante ── */
.student-box {
    background: rgba(99,202,183,0.06);
    border: 1px solid rgba(99,202,183,0.28);
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
}
.stu-avatar {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #63cab7, #2c5364);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.stu-info strong {
    display: block;
    font-family: 'Syne', sans-serif;
    font-size: 1.08rem; color: #e8e8e8; font-weight: 700;
    margin-bottom: 0.15rem;
}
.stu-info .stu-meta {
    font-size: 0.86rem; color: #8fa8b8; margin-bottom: 0.25rem;
}
.stu-info .stu-code {
    font-size: 0.84rem; color: #63cab7; font-weight: 600;
}
.stu-info a {
    color: #63cab7; text-decoration: none;
    font-size: 0.86rem; font-weight: 600;
    border-bottom: 1px dashed rgba(99,202,183,0.4);
    transition: border-color 0.2s;
}
.stu-info a:hover { border-color: #63cab7; }
 
/* ── Panel ── */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.1rem;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #63cab7;
    margin-bottom: 1.1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(99,202,183,0.18);
}
 
/* ── Inputs ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #9ab0be !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,202,183,0.18) !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
    font-family: 'Mulish', sans-serif !important;
    font-size: 0.95rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #63cab7 !important;
    box-shadow: 0 0 0 2px rgba(99,202,183,0.12) !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,202,183,0.25) !important;
    border-radius: 8px !important;
    color: #e8e8e8 !important;
}
 
/* ── Botón ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #63cab7 0%, #2c7a6e 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.87 !important; }
 
/* ── Caja resultado ── */
.result-box {
    background: linear-gradient(135deg, #0f2a24 0%, #0d1f2d 100%);
    border: 2px solid #63cab7;
    border-radius: 18px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    animation: fadeUp 0.45s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-label {
    font-size: 0.72rem; letter-spacing: 2.5px;
    text-transform: uppercase; color: #63cab7;
    margin-bottom: 0.45rem; font-weight: 700;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3.1rem; font-weight: 800;
    color: #e8f8f5; line-height: 1.1;
}
.result-model {
    font-size: 0.8rem; color: #5a8a7e; margin-top: 0.45rem;
}
 
/* ── Placeholder resultado ── */
.result-placeholder {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(99,202,183,0.2);
    border-radius: 16px;
    padding: 3rem 1.5rem;
    text-align: center;
    color: #3a5060;
}
 
/* ── Chips de métricas ── */
.metric-row {
    display: flex; gap: 0.7rem; margin-top: 1rem;
}
.metric-chip {
    flex: 1;
    background: rgba(99,202,183,0.07);
    border: 1px solid rgba(99,202,183,0.18);
    border-radius: 10px;
    padding: 0.7rem 0.4rem;
    text-align: center;
}
.m-label {
    font-size: 0.67rem; color: #4a7a6e;
    letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;
}
.m-value {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; color: #63cab7; font-weight: 700; margin-top: 0.1rem;
}
 
/* ── Expander ── */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(99,202,183,0.15) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary {
    color: #63cab7 !important; font-weight: 600 !important;
}
 
/* ── Footer ── */
.footer {
    text-align: center; padding: 1.8rem 0 0.6rem;
    color: #2a4050; font-size: 0.78rem; letter-spacing: 1px;
}
 
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>PA2: Fundamentos de Machine Learning</h1>
    <p>Predicción de Precios de Casas &nbsp;·&nbsp; Boston Housing Dataset &nbsp;·&nbsp; ISIL 2025</p>
</div>
""", unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════
# CAJA ESTUDIANTE
# ═══════════════════════════════════════════════════════════
COLAB_URL = "https://colab.research.google.com/drive/1LDE2zH2K8P1ur7Yxu-Lfb31eRIV_JSaF?usp=sharing"
 
st.markdown(f"""
<div class="student-box">
    <div class="stu-avatar">👤</div>
    <div class="stu-info">
        <strong>Luis Gabriel Beltrán Cajo</strong>
        <div class="stu-meta">📧 40468984@mail.isil.pe</div>
        <div class="stu-code">🎓 Código ISIL: 40468984</div>
        <div style="margin-top:0.35rem;">
            <a href="{COLAB_URL}" target="_blank">
                📓 Ver Cuaderno Google Colab (solo lectura) →
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════
# CARGAR MODELOS
# ═══════════════════════════════════════════════════════════
MODEL_DIR = "modelos"
 
@st.cache_resource
def load_models():
    out = {}
    for label, fname in [("Random Forest",     "random_forest.pkl"),
                         ("Gradient Boosting", "gradient_boosting.pkl")]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            out[label] = joblib.load(path)
    return out
 
models = load_models()
 
METRICAS = {
    "Random Forest":     {"R²": "0.893", "RMSE": "$3,420", "MAE": "$2,310"},
    "Gradient Boosting": {"R²": "0.917", "RMSE": "$2,980", "MAE": "$1,970"},
}
 
# ═══════════════════════════════════════════════════════════
# LAYOUT: dos columnas
# ═══════════════════════════════════════════════════════════
col_form, col_out = st.columns([1.15, 1], gap="large")
 
# ───────────────────────────────────────
# COLUMNA IZQUIERDA — Formulario
# ───────────────────────────────────────
with col_form:
    st.markdown('<div class="panel"><div class="panel-title">📐 Características de la Propiedad</div>', unsafe_allow_html=True)
 
    c1, c2 = st.columns(2)
    with c1:
        habitaciones  = st.number_input("Habitaciones",        min_value=1.0,  max_value=10.0,  value=6.0,   step=0.5,
                                         help="Nº promedio de habitaciones por vivienda")
        ratio_est     = st.number_input("Ratio Estudiantes",   min_value=5.0,  max_value=25.0,  value=15.3,  step=0.1,
                                         help="Ratio alumno/profesor por municipio")
        contaminacion = st.number_input("Contaminación",       min_value=0.0,  max_value=1.0,   value=0.54,  step=0.01,
                                         help="Concentración de óxidos nítricos (partes/10M)")
        dist_centro   = st.number_input("Dist. Centro",        min_value=0.5,  max_value=12.0,  value=3.8,   step=0.1,
                                         help="Distancia ponderada a centros de empleo")
    with c2:
        pct_bajo      = st.number_input("Pct Bajo Estatus",    min_value=0.0,  max_value=40.0,  value=12.5,  step=0.1,
                                         help="% de población de bajo estatus socioeconómico")
        tasa_crimen   = st.number_input("Tasa Crimen",         min_value=0.0,  max_value=90.0,  value=0.25,  step=0.01,
                                         help="Tasa de crimen per cápita por zona")
        antiguedad    = st.number_input("Antigüedad (%)",      min_value=0.0,  max_value=100.0, value=68.0,  step=1.0,
                                         help="% de unidades construidas antes de 1940")
        imp_propiedad = st.number_input("Impuesto Propiedad",  min_value=100.0,max_value=800.0, value=307.0, step=1.0,
                                         help="Tasa de impuesto a la propiedad por $10,000")
 
    st.markdown('</div>', unsafe_allow_html=True)
 
    # Selector de modelo
    st.markdown('<div class="panel"><div class="panel-title">🤖 Modelo de Predicción</div>', unsafe_allow_html=True)
    modelo_sel = st.selectbox("Elige el modelo", options=list(METRICAS.keys()), label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
 
    predecir = st.button("🔍  Predecir Precio", use_container_width=True)
 
# ───────────────────────────────────────
# COLUMNA DERECHA — Resultado
# ───────────────────────────────────────
with col_out:
 
    st.markdown('<div class="panel-title" style="font-family:\'Syne\',sans-serif;font-size:0.73rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#63cab7;margin-bottom:1rem;">📊 Resultado</div>', unsafe_allow_html=True)
 
    if predecir:
        log_crimen = np.log1p(tasa_crimen)
        X = np.array([[habitaciones, pct_bajo, ratio_est,
                       log_crimen, contaminacion, antiguedad,
                       dist_centro, imp_propiedad]])
 
        if modelo_sel not in models:
            st.error(
                f"⚠️ Modelo **{modelo_sel}** no encontrado en `modelos/`.\n\n"
                "Verifica que los archivos `.pkl` estén subidos a GitHub en la carpeta `modelos/`."
            )
        else:
            precio = models[modelo_sel].predict(X)[0]
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Precio Estimado de la Vivienda</div>
                <div class="result-price">${precio * 1000:,.0f}</div>
                <div class="result-model">Modelo: {modelo_sel} &nbsp;·&nbsp; USD</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-placeholder">
            <div style="font-size:2.8rem;margin-bottom:0.7rem;">🏠</div>
            <div style="font-size:0.88rem;">
                Completa los campos y presiona<br>
                <strong style="color:#63cab7;">Predecir Precio</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    # Chips de métricas
    m = METRICAS[modelo_sel]
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip">
            <div class="m-label">R²</div>
            <div class="m-value">{m['R²']}</div>
        </div>
        <div class="metric-chip">
            <div class="m-label">RMSE</div>
            <div class="m-value">{m['RMSE']}</div>
        </div>
        <div class="metric-chip">
            <div class="m-label">MAE</div>
            <div class="m-value">{m['MAE']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Expander ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Descripción del dataset y del proyecto"):
        st.markdown(f"""
**Dataset:** Boston Housing Dataset  
**Registros:** 506 &nbsp;|&nbsp; **Variables originales:** 14  
**Fuente:** Harrison & Rubinfeld (1978) · UCI ML Repository
 
**Objetivo del modelo:** Predecir el precio mediano de viviendas (en miles de USD)
a partir de características urbanas, ambientales y sociales del vecindario.
 
---
**Variables utilizadas:**
 
| Variable | Descripción |
|---|---|
| Habitaciones | Promedio de habitaciones por vivienda |
| Pct Bajo Estatus | % población de bajo estatus socioeconómico |
| Ratio Estudiantes | Ratio alumno / profesor por municipio |
| Tasa Crimen | Crimen per cápita (se aplica log-transform) |
| Contaminación | Óxidos nítricos en partes por 10 millones |
| Antigüedad | % de unidades construidas antes de 1940 |
| Dist. Centro | Distancia ponderada a centros de empleo |
| Impuesto Propiedad | Tasa de impuesto por cada $10,000 |
 
---
**Modelos entrenados:**
 
- 🌲 **Random Forest** — R² test: 0.893
- 🚀 **Gradient Boosting** — R² test: 0.917 *(mejor modelo)*
 
**Preprocesamiento:** Outliers tratados con IQR clipping · Log-transform en TasaCrimen  
**División:** 80/20 con `stratify` por quintiles de precio · Validación cruzada 5-fold
 
---
**Autor:** Luis Gabriel Beltrán Cajo &nbsp;|&nbsp; **Código:** 40468984  
**Correo:** 40468984@mail.isil.pe  
📓 [Cuaderno Colab (solo lectura)]({COLAB_URL})
        """)
 
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PA2 &nbsp;·&nbsp; Fundamentos de Machine Learning &nbsp;·&nbsp; ISIL 2025
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Luis Gabriel Beltrán Cajo &nbsp;·&nbsp; 40468984
</div>
""", unsafe_allow_html=True)
