# app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(page_title="Evaluación nutricional", layout="wide")

ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
MODEL_PATH = ART_DIR / "model.joblib"
METRICS_PATH = ART_DIR / "metrics.json"


# =========================
# Estilo simple
# =========================
import base64
from pathlib import Path
import streamlit as st


def add_bg_watermark(image_path: str, opacity: float = 0.05, size_px: int = 900):
    # Lee la imagen y la convierte a base64
    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Inyecta CSS (f-string) con el base64 y la opacidad
    st.markdown(
        f"""
        <style>
        /* ===== ESTILO GENERAL ===== */
        h1, h2, h3 {{
            color: #1E3A8A;
        }}

        .card {{
            background: white;
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            box-shadow: 0 8px 20px rgba(20, 60, 120, 0.08);
            border: 1px solid #E3ECFA;
            margin-bottom: 1.5rem;
        }}

        .muted {{
            color: #475569;
            font-size: 0.9rem;
        }}

        div.stButton > button {{
            background-color: #1E40AF;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: #1D4ED8;
        }}

        /* ===== WATERMARK ===== */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:image/png;base64,{b64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: {size_px}px auto;
            opacity: {opacity};
            z-index: 0;
            pointer-events: none;
        }}

        /* Todo el contenido por encima del watermark */
        section[data-testid="stAppViewContainer"] {{
            position: relative;
            z-index: 1;
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Llama la función UNA VEZ, idealmente al inicio del script
add_bg_watermark("data/watermark.png", opacity=0.5, size_px=900)


# =========================
# Carga artifacts
# =========================
@st.cache_resource(show_spinner=False)
def load_model_payload(path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Retorna (model_pipeline, payload_dict)
    - Si model.joblib es un dict con key 'model', extrae ese modelo.
    - Si es un modelo directo, payload queda vacío.
    """
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"], obj
    return obj, {}


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def pastel_scale():
    return [
        [0.0, "#FBFCFF"],
        [0.2, "#F2F6FF"],
        [0.4, "#E8F0FF"],
        [0.6, "#DCE9FF"],
        [0.8, "#CFE0FF"],
        [1.0, "#BED3FF"],
    ]


def parse_float(txt: str) -> Optional[float]:
    txt = (txt or "").strip().replace(",", ".")
    if txt == "":
        return None
    try:
        return float(txt)
    except Exception:
        return None


def parse_int(txt: str) -> Optional[int]:
    v = parse_float(txt)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def extract_accuracy(metrics: Dict[str, Any]) -> Optional[float]:
    for k in ["accuracy", "accuracy_test", "test_accuracy"]:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def extract_report(metrics: Dict[str, Any]) -> Optional[pd.DataFrame]:
    rep = metrics.get("report")
    if isinstance(rep, dict):
        try:
            df = pd.DataFrame(rep).T
            return df
        except Exception:
            return None

    rep2 = metrics.get("classification_report")
    if isinstance(rep2, dict):
        try:
            df = pd.DataFrame(rep2).T
            return df
        except Exception:
            return None

    return None


def infer_labels_from_report(df_report: pd.DataFrame) -> List[str]:
    # Quita filas típicas de promedios si existen
    bad = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    labels = [str(i) for i in df_report.index if str(i) not in bad]
    return labels


def extract_confusion(metrics: Dict[str, Any]) -> Optional[np.ndarray]:
    cm = metrics.get("confusion_matrix")
    if cm is None:
        return None
    try:
        arr = np.array(cm, dtype=int)
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def infer_confusion_labels(metrics: Dict[str, Any], model: Any) -> Optional[List[str]]:
    # 1) class_labels explícitos
    cl = metrics.get("class_labels")
    if isinstance(cl, list) and len(cl) > 0:
        return [str(x).replace("_", " ") for x in cl]

    # 2) desde reporte
    df_rep = extract_report(metrics)
    if df_rep is not None and len(df_rep.index) > 0:
        return [x.replace("_", " ") for x in infer_labels_from_report(df_rep)]

    # 3) desde model.classes_
    if hasattr(model, "classes_"):
        try:
            return [str(x).replace("_", " ") for x in list(model.classes_)]
        except Exception:
            pass

    return None


def make_confusion_heatmap(cm: np.ndarray, labels: List[str]):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # suaviza zmax para que no quede muy oscuro si hay una clase muy dominante
    zmax_soft = float(np.quantile(df_cm.values, 0.95)) if df_cm.values.size else 1.0

    fig = px.imshow(
        df_cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=pastel_scale(),
        zmin=0,
        zmax=zmax_soft,
        labels={"x": "Predicho", "y": "Real", "color": "Casos"},
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=520)
    return fig


# =========================
# Arranque
# =========================
st.markdown(
    """
    <h1>Evaluación Nutricional Predictiva</h1>
    <p class="muted">
    Sistema de clasificación basado en hábitos, características físicas y antecedentes.
    </p>
    """,
    unsafe_allow_html=True
)

if not MODEL_PATH.exists():
    st.error("No se encontró artifacts/model.joblib.")
    st.stop()

model, payload = load_model_payload(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

# Columnas esperadas (si el payload las trae, se usan para ordenar)
num_features = payload.get("num_features", []) if isinstance(payload, dict) else []
cat_features = payload.get("cat_features", []) if isinstance(payload, dict) else []
expected_cols = (list(num_features) + list(cat_features)) if (num_features or cat_features) else None

tabs = st.tabs(["Resultados", "Probar el modelo"])


# =========================
# TAB: Resultados (orden + pastel)
# =========================
with tabs[0]:
    st.subheader("Resultados del clasificador")

    acc = extract_accuracy(metrics)
    df_rep = extract_report(metrics)
    cm = extract_confusion(metrics)
    cm_labels = infer_confusion_labels(metrics, model) if cm is not None else None

    # 1) Accuracy + Reporte
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.markdown("**Accuracy (test)**")
        if acc is None:
            st.markdown('<div class="muted">No disponible.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='font-size:46px; font-weight:750; line-height:1.05;'>{acc:.4f}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("**Reporte por clase**")
        if df_rep is None or df_rep.empty:
            st.info("Reporte por clase no disponible.")
        else:
            df_show = df_rep.copy()
            # Formato estándar si existen estas columnas
            cols_pref = [c for c in ["precision", "recall", "f1-score", "support"] if c in df_show.columns]
            if cols_pref:
                df_show = df_show[cols_pref]
            for c in df_show.columns:
                if c != "support":
                    df_show[c] = pd.to_numeric(df_show[c], errors="coerce")
            st.dataframe(
                df_show.style.format({c: "{:.4f}" for c in df_show.columns if c != "support"} | {"support": "{:.0f}"}),
                use_container_width=True,
                hide_index=False,
            )

    # 2) Matriz de confusión (al final)
    st.markdown("---")
    st.markdown("### Matriz de confusión")

    if cm is None:
        st.info("Matriz de confusión no disponible.")
    else:
        if cm_labels is None:
            # Si no se pueden inferir etiquetas, al menos mostramos con índices
            cm_labels = [f"Clase {i+1}" for i in range(cm.shape[0])]
        fig_cm = make_confusion_heatmap(cm, cm_labels)
        st.plotly_chart(fig_cm, use_container_width=True)


# =========================
# TAB: Probar el modelo
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
with tabs[1]:
    st.subheader("Prueba del clasificador")

    st.write("Ingrese los datos y presione **Clasificar**.")

    # Layout más angosto y “clínico”
    _, center, _ = st.columns([1, 2.2, 1])

    # Labels humanos (si tu modelo está en esquema “A” original o “B” español)
    # Si tu payload trae cat_choices, los usamos para poblar selects.
    cat_choices = payload.get("cat_choices", {}) if isinstance(payload, dict) else {}

    def cat_options(name: str, fallback: List[str]) -> List[str]:
        opts = cat_choices.get(name)
        if isinstance(opts, list) and len(opts) > 0:
            return opts
        return fallback

    # Definimos un set mínimo que funciona para ambos esquemas:
    # - Si expected_cols existe, se respetará ese orden al final.
    # - Si no, usamos columnas comunes más probables.
    # (No inventamos escalas; todo numérico se ingresa como texto numérico.)
    # Labels humanos para campos comunes:
    HUMAN = {
        "Age": "Edad (años)",
        "Height": "Estatura (m)",
        "Weight": "Peso (kg)",
        "FCVC": "Consumo de vegetales",
        "NCP": "Comidas principales al día",
        "CH2O": "Consumo de agua",
        "FAF": "Actividad física",
        "TUE": "Tiempo frente a pantallas (horas/día)",
        "Gender": "Género",
        "family_history_with_overweight": "Historia familiar de sobrepeso",
        "FAVC": "Consumo frecuente de comida hipercalórica",
        "SMOKE": "Fuma",
        "SCC": "Monitorea calorías",
        "MTRANS": "Medio de transporte habitual",
        # Alternativos en español si tu modelo usa esos nombres:
        "edad": "Edad (años)",
        "estatura_m": "Estatura (m)",
        "peso_kg": "Peso (kg)",
        "consumo_vegetales_ord": "Consumo de vegetales",
        "numero_comidas_principales_ord": "Comidas principales al día",
        "consumo_agua_ord": "Consumo de agua",
        "actividad_fisica_ord": "Actividad física",
        "tiempo_uso_tecnologia_ord": "Tiempo frente a pantallas (horas/día)",
        "genero": "Género",
        "historia_familiar_sobrepeso": "Historia familiar de sobrepeso",
        "consumo_comida_hipercalorica": "Consumo frecuente de comida hipercalórica",
        "fuma": "Fuma",
        "monitorea_calorias": "Monitorea calorías",
        "medio_transporte": "Medio de transporte habitual",
    }

    # Campos por defecto si no hay expected_cols
    fallback_num = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    fallback_cat = ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC", "MTRANS"]

    num_cols_ui = list(num_features) if num_features else fallback_num
    cat_cols_ui = list(cat_features) if cat_features else fallback_cat

    with center:
        with st.form("form_case"):
            st.markdown("### Datos generales")
            a, b, c = st.columns(3, gap="large")
            age_txt = a.text_input(HUMAN.get(num_cols_ui[0], num_cols_ui[0]), value="", placeholder="Ej: 25")
            h_txt = b.text_input(HUMAN.get(num_cols_ui[1], num_cols_ui[1]), value="", placeholder="Ej: 1.70")
            w_txt = c.text_input(HUMAN.get(num_cols_ui[2], num_cols_ui[2]), value="", placeholder="Ej: 70")

            st.markdown("---")
            st.markdown("### Hábitos")
            d, e, f = st.columns(3, gap="large")
            n4_txt = d.text_input(HUMAN.get(num_cols_ui[3], num_cols_ui[3]), value="", placeholder="Ej: 2")
            n5_txt = e.text_input(HUMAN.get(num_cols_ui[4], num_cols_ui[4]), value="", placeholder="Ej: 3")
            n6_txt = f.text_input(HUMAN.get(num_cols_ui[5], num_cols_ui[5]), value="", placeholder="Ej: 2")

            g, h = st.columns(2, gap="large")
            n7_txt = g.text_input(HUMAN.get(num_cols_ui[6], num_cols_ui[6]), value="", placeholder="Ej: 1")
            n8_txt = h.text_input(HUMAN.get(num_cols_ui[7], num_cols_ui[7]), value="", placeholder="Ej: 1")

            st.write("")
            i, j = st.columns(2, gap="large")
            favc = i.selectbox(HUMAN.get(cat_cols_ui[2], cat_cols_ui[2]), options=cat_options(cat_cols_ui[2], ["No", "Sí"]), index=0)
            mtrans = j.selectbox(HUMAN.get(cat_cols_ui[5], cat_cols_ui[5]), options=cat_options(cat_cols_ui[5], ["Automóvil", "Transporte público", "Bicicleta", "Caminando"]), index=0)

            st.markdown("---")
            st.markdown("### Antecedentes")
            k, l, m = st.columns(3, gap="large")
            gender = k.selectbox(HUMAN.get(cat_cols_ui[0], cat_cols_ui[0]), options=cat_options(cat_cols_ui[0], ["Hombre", "Mujer"]), index=0)
            fam = l.selectbox(HUMAN.get(cat_cols_ui[1], cat_cols_ui[1]), options=cat_options(cat_cols_ui[1], ["No", "Sí"]), index=0)
            smoke = m.selectbox(HUMAN.get(cat_cols_ui[3], cat_cols_ui[3]), options=cat_options(cat_cols_ui[3], ["No", "Sí"]), index=0)

            n, _ = st.columns(2, gap="large")
            scc = n.selectbox(HUMAN.get(cat_cols_ui[4], cat_cols_ui[4]), options=cat_options(cat_cols_ui[4], ["No", "Sí"]), index=0)

            submitted = st.form_submit_button("Clasificar")

        if submitted:
            # Parse numéricos
            n_age = parse_float(age_txt)
            n_h = parse_float(h_txt)
            n_w = parse_float(w_txt)
            n4 = parse_float(n4_txt)
            n5 = parse_float(n5_txt)
            n6 = parse_float(n6_txt)
            n7 = parse_float(n7_txt)
            n8 = parse_float(n8_txt)

            missing = []
            if n_age is None: missing.append(HUMAN.get(num_cols_ui[0], num_cols_ui[0]))
            if n_h is None: missing.append(HUMAN.get(num_cols_ui[1], num_cols_ui[1]))
            if n_w is None: missing.append(HUMAN.get(num_cols_ui[2], num_cols_ui[2]))
            if n4 is None: missing.append(HUMAN.get(num_cols_ui[3], num_cols_ui[3]))
            if n5 is None: missing.append(HUMAN.get(num_cols_ui[4], num_cols_ui[4]))
            if n6 is None: missing.append(HUMAN.get(num_cols_ui[5], num_cols_ui[5]))
            if n7 is None: missing.append(HUMAN.get(num_cols_ui[6], num_cols_ui[6]))
            if n8 is None: missing.append(HUMAN.get(num_cols_ui[7], num_cols_ui[7]))

            if missing:
                st.error("Revisa estos campos numéricos: " + ", ".join(missing))
            else:
                row = {
                    num_cols_ui[0]: n_age,
                    num_cols_ui[1]: n_h,
                    num_cols_ui[2]: n_w,
                    num_cols_ui[3]: n4,
                    num_cols_ui[4]: n5,
                    num_cols_ui[5]: n6,
                    num_cols_ui[6]: n7,
                    num_cols_ui[7]: n8,
                    cat_cols_ui[0]: gender,
                    cat_cols_ui[1]: fam,
                    cat_cols_ui[2]: favc,
                    cat_cols_ui[3]: smoke,
                    cat_cols_ui[4]: scc,
                    cat_cols_ui[5]: mtrans,
                }

                df = pd.DataFrame([row])

                # Orden esperado por el modelo (si está disponible en el payload)
                if expected_cols:
                    for c in expected_cols:
                        if c not in df.columns:
                            df[c] = np.nan
                    df = df[expected_cols]

                try:
                    pred = model.predict(df)[0]
                    st.success(f"Resultado del modelo: {str(pred).replace('_', ' ')}")
                except Exception as e:
                    st.error(f"No se pudo clasificar con los datos ingresados. Detalle: {e}")
st.markdown('</div>', unsafe_allow_html=True)
