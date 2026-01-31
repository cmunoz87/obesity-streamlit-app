# train_model.py
# Entrena el MISMO clasificador que usaste en el notebook (pipeline + RandomForest)
# y exporta artifacts/model.joblib + artifacts/metrics.json

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

RENAME_COLUMNS = {
    "Gender": "genero",
    "Age": "edad",
    "Height": "estatura_m",
    "Weight": "peso_kg",
    "family_history_with_overweight": "historia_familiar_sobrepeso",
    "FAVC": "consumo_comida_hipercalorica",
    "FCVC": "consumo_vegetales",
    "NCP": "numero_comidas_principales",
    "CAEC": "consumo_entre_comidas",
    "SMOKE": "fuma",
    "CH2O": "consumo_agua",
    "SCC": "monitorea_calorias",
    "FAF": "actividad_fisica",
    "TUE": "tiempo_uso_tecnologia",
    "CALC": "consumo_alcohol",
    "MTRANS": "medio_transporte",
    "NObeyesdad": "nivel_obesidad",
}

VALUE_MAPS = {
    "genero": {"Female": "Mujer", "Male": "Hombre"},
    "historia_familiar_sobrepeso": {"yes": "Si", "no": "No"},
    "consumo_comida_hipercalorica": {"yes": "Si", "no": "No"},
    "fuma": {"yes": "Si", "no": "No"},
    "monitorea_calorias": {"yes": "Si", "no": "No"},
    "consumo_entre_comidas": {"no": "Nunca", "Sometimes": "A_veces", "Frequently": "Frecuente", "Always": "Siempre"},
    "consumo_alcohol": {"no": "Nunca", "Sometimes": "A_veces", "Frequently": "Frecuente", "Always": "Siempre"},
    "medio_transporte": {
        "Public_Transportation": "Transporte_publico",
        "Walking": "Caminando",
        "Automobile": "Automovil",
        "Motorbike": "Motocicleta",
        "Bike": "Bicicleta",
    },
    "nivel_obesidad": {
        "Insufficient_Weight": "Bajo_peso",
        "Normal_Weight": "Peso_normal",
        "Overweight_Level_I": "Sobrepeso_I",
        "Overweight_Level_II": "Sobrepeso_II",
        "Obesity_Type_I": "Obesidad_I",
        "Obesity_Type_II": "Obesidad_II",
        "Obesity_Type_III": "Obesidad_III",
    },
}

# En tu notebook trabajaste estas como "ordinales" (_ord).
# Aquí las construimos redondeando y acotando a rangos típicos (según tu diagnóstico).
ORDINAL_SPECS = {
    "consumo_vegetales": (1, 3),
    "numero_comidas_principales": (1, 4),
    "consumo_agua": (1, 3),
    "actividad_fisica": (0, 3),
    "tiempo_uso_tecnologia": (0, 2),
}

ORDER_NIVEL_OBESIDAD = [
    "Bajo_peso",
    "Peso_normal",
    "Sobrepeso_I",
    "Sobrepeso_II",
    "Obesidad_I",
    "Obesidad_II",
    "Obesidad_III",
]

NUM_FEATURES = [
    "edad",
    "estatura_m",
    "peso_kg",
    "consumo_vegetales_ord",
    "numero_comidas_principales_ord",
    "consumo_agua_ord",
    "actividad_fisica_ord",
    "tiempo_uso_tecnologia_ord",
]

CAT_FEATURES = [
    "genero",
    "historia_familiar_sobrepeso",
    "consumo_comida_hipercalorica",
    "fuma",
    "monitorea_calorias",
    "medio_transporte",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_COLUMNS)

    for col, mapping in VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Orden del target (como en tu notebook)
    df["nivel_obesidad"] = pd.Categorical(
        df["nivel_obesidad"], categories=ORDER_NIVEL_OBESIDAD, ordered=True
    )

    # Crear variables ordinales _ord
    for col, (vmin, vmax) in ORDINAL_SPECS.items():
        if col not in df.columns:
            raise ValueError(f"Falta columna esperada: {col}")
        df[f"{col}_ord"] = (
            pd.to_numeric(df[col], errors="coerce")
            .round()
            .clip(vmin, vmax)
            .astype(int)
        )

    return df


def main():
    csv_path = os.path.join("data", "ObesityDataSet_raw_and_data_sinthetic.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el CSV en {csv_path}. "
            "Pon el archivo dentro de data/ con ese nombre."
        )

    df = load_dataset(csv_path)

    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    y = df["nivel_obesidad"].astype(str).copy()

    # Split igual que en tu notebook: test_size=0.25, stratify, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )

    modelo_rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    modelo_rf.fit(X_train, y_train)

    y_pred = modelo_rf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = ORDER_NIVEL_OBESIDAD
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Guardar artefactos
    os.makedirs("artifacts", exist_ok=True)

    cat_choices = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_FEATURES}

    payload = {
        "model": modelo_rf,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "cat_choices": cat_choices,
        "labels": labels,
    }
    joblib.dump(payload, os.path.join("artifacts", "model.joblib"))

    metrics = {
        "accuracy": acc,
        "classification_report": report_dict,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join("artifacts", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("OK - Artefactos generados:")
    print("- artifacts/model.joblib")
    print("- artifacts/metrics.json")
    print(f"Accuracy (test): {acc:.4f}")


if __name__ == "__main__":
    main()
