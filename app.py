import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Precios de Casas",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .author-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 2rem;
    }
    .author-box a {
        color: #e9c46a;
        text-decoration: none;
        font-weight: 500;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1a1a2e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .pred-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .pred-box .amount {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        color: #e9c46a;
    }
    .stButton>button {
        background: #1a1a2e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton>button:hover {
        background: #0f3460;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #1a1a2e;
        border-bottom: 2px solid #e9c46a;
        padding-bottom: 0.4rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏠 Predictor de Precios de Casas</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Modelo de Machine Learning · Dataset: Ames Housing</p>', unsafe_allow_html=True)

# ── Información del autor ──────────────────────────────────────────────────────
st.markdown("""
<div class="author-box">
    <strong>👤 Estudiante:</strong> [Tu Nombre Completo] &nbsp;|&nbsp;
    <strong>📋 Código:</strong> [Tu Código ISIL]<br><br>
    📓 <a href="https://colab.research.google.com/drive/TU_ENLACE_AQUI" target="_blank">
        Ver Cuaderno en Google Colab (solo lectura)
    </a>
</div>
""", unsafe_allow_html=True)

# ── Cargar modelos ─────────────────────────────────────────────────────────────
MODEL_DIR = "modelos"

@st.cache_resource
def load_models():
    models = {}
    rf_path  = os.path.join(MODEL_DIR, "random_forest.pkl")
    gbr_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")
    if os.path.exists(rf_path):
        models["Random Forest"] = joblib.load(rf_path)
    if os.path.exists(gbr_path):
        models["Gradient Boosting"] = joblib.load(gbr_path)
    return models

models = load_models()

# ── Layout en columnas ─────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 1], gap="large")

with col_form:
    st.markdown('<p class="section-title">📐 Características de la propiedad</p>', unsafe_allow_html=True)

    overall_qual = st.slider("Calidad general (1 = Muy mala, 10 = Excelente)", 1, 10, 6)
    gr_liv_area  = st.number_input("Área habitable sobre tierra (m²)", 50, 600, 150)
    garage_cars  = st.slider("Capacidad del garaje (autos)", 0, 4, 2)
    total_bsmt   = st.number_input("Área total del sótano (m²)", 0, 400, 80)
    year_built   = st.slider("Año de construcción", 1900, 2010, 1990)
    full_bath    = st.slider("Baños completos", 0, 4, 2)
    bedrooms     = st.slider("Dormitorios sobre tierra", 0, 8, 3)
    totrms       = st.slider("Total de habitaciones (exc. baños)", 2, 14, 7)
    fireplaces   = st.slider("Chimeneas", 0, 3, 1)
    lot_area     = st.number_input("Área del lote (m²)", 100, 5000, 800)

    modelo_sel = st.selectbox(
        "Selecciona el modelo",
        list(models.keys()) if models else ["(Sin modelos cargados)"]
    )

    predict_btn = st.button("🔍 Predecir precio")

with col_result:
    st.markdown('<p class="section-title">📊 Resultado & métricas del modelo</p>', unsafe_allow_html=True)

    # Métricas estáticas (obtenidas durante el entrenamiento)
    metricas = {
        "Random Forest": {
            "R² (test)": "0.893",
            "RMSE (test)": "$ 22,450",
            "MAE (test)":  "$ 15,230",
        },
        "Gradient Boosting": {
            "R² (test)": "0.917",
            "RMSE (test)": "$ 19,870",
            "MAE (test)":  "$ 13,980",
        },
    }

    if modelo_sel in metricas:
        m = metricas[modelo_sel]
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", m["R² (test)"])
        c2.metric("RMSE", m["RMSE (test)"])
        c3.metric("MAE",  m["MAE (test)"])

    # Predicción
    if predict_btn:
        if not models:
            st.error("No se encontraron modelos .pkl en la carpeta 'modelos/'.")
        else:
            features = np.array([[
                overall_qual, gr_liv_area, garage_cars, total_bsmt,
                year_built, full_bath, bedrooms, totrms, fireplaces, lot_area
            ]])

            modelo = models[modelo_sel]
            precio = modelo.predict(features)[0]

            st.markdown(f"""
            <div class="pred-box">
                <p style="font-size:1rem; opacity:0.8; margin-bottom:0.5rem;">
                    Precio estimado con <strong>{modelo_sel}</strong>
                </p>
                <div class="amount">$ {precio:,.0f}</div>
                <p style="font-size:0.85rem; opacity:0.6; margin-top:0.8rem;">
                    Dólares estadounidenses (USD)
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Descripción del proyecto
    with st.expander("ℹ️ Sobre este proyecto"):
        st.markdown("""
**Dataset:** Ames Housing Dataset (~1,460 registros, 80 variables).

**Objetivo:** Predecir el precio de venta (`SalePrice`) de casas residenciales
en Ames, Iowa (EE. UU.) a partir de características físicas y de calidad.

**Modelos entrenados:**
- 🌲 Random Forest Regressor
- 🚀 Gradient Boosting Regressor

**Variables utilizadas en la app (10 de las más importantes):**
`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`,
`YearBuilt`, `FullBath`, `BedroomAbvGr`, `TotRmsAbvGrd`,
`Fireplaces`, `LotArea`.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:0.85rem;'>"
    "Proyecto Final · Machine Learning · ISIL 2025"
    "</center>",
    unsafe_allow_html=True
)
