# Clasificador de Obesidad (Solemne 1) - Streamlit

App en Streamlit para:
- Visualizar resultados del clasificador (accuracy, reporte por clase, matriz de confusión)
- Probar el modelo con datos ingresados por pantalla

## 1) Ejecutar local

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

python train_model.py
streamlit run app.py
```

## 2) Deploy en Streamlit Community Cloud

1. Sube este repo a GitHub (tal cual).
2. En Streamlit Community Cloud:
   - New app
   - selecciona tu repo
   - Main file path: `app.py`
   - Deploy

Notas:
- El modelo se carga desde `artifacts/model.joblib`.
- Las métricas/CM se leen de `artifacts/metrics.json`.
