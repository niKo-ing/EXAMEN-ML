# 🏦 Home Credit Default Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Evaluación de Riesgo Crediticio utilizando Machine Learning y Metodología CRISP-DM.**

Este proyecto implementa una solución *end-to-end* para predecir la probabilidad de incumplimiento de pago (default) de clientes, utilizando el dataset de **Home Credit Default Risk**. La solución está estructurada modularmente siguiendo las fases de CRISP-DM y despliega un modelo productivo a través de una API REST.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Metodología y Enfoque Técnico](#-metodología-y-enfoque-técnico)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Ejecución](#-ejecución)
- [Notebook Consolidado (Examen)](#-notebook-consolidado)
- [API de Predicción](#-api-de-predicción)
- [Resultados](#-resultados)

---

## 📖 Descripción del Proyecto

El objetivo es facilitar la toma de decisiones financieras mediante la automatización de la evaluación de riesgo. El sistema analiza múltiples fuentes de datos (historial de buró, pagos previos, saldos de tarjetas, etc.) para calcular una probabilidad de impago y sugerir una decisión (Aprobar, Revisión Manual, Rechazar).

### Características Clave
- **Integración Multi-Fuente**: Fusión de 7 tablas relacionales (Bureau, POS Cash, Installments, etc.).
- **Ingeniería de Características Avanzada**: Agregaciones estadísticas, manejo de categorías y optimización de memoria.
- **Modelo Robusto**: Uso de LightGBM con manejo explícito de desbalance de clases (`is_unbalance=True`).
- **Despliegue API**: Endpoint `/evaluate_risk` listo para integración en sistemas bancarios.

---

## 📂 Estructura del Repositorio

El proyecto sigue una estructura modular donde cada directorio representa una fase del ciclo de vida de ML:

```bash
EXAMEN-ML/
├── 01_data_understanding/   # EDA y análisis exploratorio
│   ├── EDA.ipynb           # Notebook detallado de exploración
│   └── eda.py              # Script auxiliar
├── 02_data_preparation/     # Procesamiento de datos
│   └── feature_engineering.py # Pipeline de limpieza y agregación
├── 03_modeling/             # Entrenamiento de modelos
│   └── train.py            # Entrenamiento LightGBM y guardado de artefactos
├── 04_evaluation/           # Validación
│   └── evaluate_model.py   # Generación de métricas y gráficos ROC/AUC
├── 05_deployment/           # Servicio API
│   └── app.py              # Aplicación FastAPI
├── 06_TODO_EN_IPYNB/        # ⭐️ PARA REVISIÓN RÁPIDA
│   └── Examen_Completo.ipynb # Todo el proyecto en un solo notebook
├── artifacts/               # Modelos serializados y gráficos
│   └── plots/              # Curvas ROC, Matriz de Confusión, Feature Importance
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación
```

---

## 🛠 Metodología y Enfoque Técnico

1.  **Entendimiento de Datos**: Análisis de distribución de clases (fuerte desbalance detectado), correlaciones y calidad de datos.
2.  **Ingeniería de Características**:
    - Casting inteligente de tipos para reducción de memoria (`reduce_mem_usage`).
    - One-Hot Encoding para variables categóricas.
    - Agregaciones (Mean, Max, Min, Sum) para tablas transaccionales (Bureau, Installments).
3.  **Modelado**:
    - Algoritmo: **LightGBM**.
    - Estrategia de Desbalance: `is_unbalance=True` (peso automático a la clase minoritaria).
    - Métricas de Optimización: AUC-ROC.
4.  **Despliegue**:
    - Framework: **FastAPI**.
    - Validación de Entrada: Pydantic models.
    - Sanitización: Limpieza de nombres de features JSON-incompatibles.

---

## 🚀 Instalación y Configuración

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/niKo-ing/EXAMEN-ML.git
    cd EXAMEN-ML
    ```

2.  **Crear entorno virtual (Opcional pero recomendado)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Ejecución

Puedes ejecutar el proyecto paso a paso mediante scripts modulares:

1.  **Procesar Datos**:
    ```bash
    python 02_data_preparation/feature_engineering.py
    ```
2.  **Entrenar Modelo**:
    ```bash
    python 03_modeling/train.py
    ```
3.  **Evaluar**:
    ```bash
    python 04_evaluation/evaluate_model.py
    ```
4.  **Iniciar API**:
    ```bash
    cd 05_deployment
    uvicorn app:app --reload
    ```

---

## 📓 Notebook Consolidado

Para facilitar la revisión académica o demostraciones rápidas, se ha consolidado todo el flujo de trabajo en un único archivo Jupyter Notebook.

📍 **Ubicación**: `06_TODO_EN_IPYNB/Examen_Completo.ipynb`

Este notebook incluye:
- Carga y limpieza de datos.
- Entrenamiento del modelo.
- Visualización de resultados.
- Simulación de peticiones a la API.

---

## 📡 API de Predicción

Una vez iniciada la API (paso 4 de Ejecución), puedes probarla localmente.

- **URL Base**: `http://localhost:8000`
- **Documentación Swagger UI**: `http://localhost:8000/docs`

### Ejemplo de Request (`POST /evaluate_risk`)

```json
{
  "features": {
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4,
    "AMT_CREDIT": 100000,
    "AMT_ANNUITY": 5000,
    "CODE_GENDER": "M",
    ...
  }
}
```

### Ejemplo de Response

```json
{
  "default_probability": 0.045,
  "decision": "APROBAR",
  "risk_level": "Bajo",
  "model_version": "1.0.0"
}
```

---

## 📊 Resultados

El modelo ha sido evaluado utilizando validación cruzada y métricas robustas para problemas desbalanceados.

- **AUC-ROC**: *Disponible en artifacts/plots/roc_curve.png*
- **Feature Importance**: Las variables más influyentes suelen ser fuentes externas (`EXT_SOURCE`) y edad (`DAYS_BIRTH`).

---

**Autor**: [Tu Nombre/Usuario]
**Examen de Machine Learning**
