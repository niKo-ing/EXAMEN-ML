# Predicción de Riesgo de Incumplimiento de Crédito (Home Credit Default Risk)

Este proyecto desarrolla un modelo de Machine Learning para predecir el riesgo de impago de créditos, siguiendo la metodología CRISP-DM y estructurado como microservicios.

## Estructura del Proyecto

- `/01_data_understanding`: 
    - `EDA.ipynb`: Notebook con análisis exploratorio de datos, visualización de distribuciones y valores nulos.
    - `eda.py`: Script auxiliar de exploración.
- `/02_data_preparation`: 
    - `feature_engineering.py`: Script principal que carga, limpia, agrega y une todas las tablas (Bureau, POS_CASH, etc.) con la tabla principal. Genera `processed_data.parquet`.
- `/03_modeling`: 
    - `train.py`: Entrena un modelo LightGBM con manejo de desbalance de clases. Guarda el modelo (`lgbm_model.pkl`) y la lista de características (`features.joblib`) en `/artifacts`.
- `/04_evaluation`: 
    - `evaluate.py`: Scripts para evaluación adicional (pendiente de expansión).
- `/05_deployment`: 
    - `app.py`: API REST construida con FastAPI para servir el modelo.
- `/06_TODO_EN_IPYNB`:
    - `Examen_Completo.ipynb`: Notebook consolidado que contiene todo el flujo del proyecto (Carga, Entrenamiento, Evaluación, Simulación API).

## Requisitos

Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Instrucciones de Ejecución

1.  **Preparación de Datos**:
    Ejecutar el script de ingeniería de características para generar el dataset de entrenamiento.
    ```bash
    python 02_data_preparation/feature_engineering.py
    ```

2.  **Entrenamiento del Modelo**:
    Entrenar el modelo LightGBM.
    ```bash
    python 03_modeling/train.py
    ```

3.  **Evaluación del Modelo**:
    Generar gráficos de rendimiento y métricas.
    ```bash
    python 04_evaluation/evaluate_model.py
    ```

4.  **Despliegue de la API**:
    Iniciar el servidor de la API.
    ```bash
    cd 05_deployment
    uvicorn app:app --reload
    ```
    La API estará disponible en `http://localhost:8000`.
    Documentación interactiva en `http://localhost:8000/docs`.

## Ejemplo de Uso de la API

Endpoint: `POST /evaluate_risk`

Body (JSON):
```json
{
  "data": {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4,
    "AMT_CREDIT": 100000,
    "AMT_ANNUITY": 5000,
    ... (otras características)
  }
}
```

Respuesta:
```json
{
  "default_probability": 0.05,
  "decision": "APROBAR",
  "risk_level": "Low"
}
```
