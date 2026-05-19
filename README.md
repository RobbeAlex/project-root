# Proyecto MLOps: Predicción de Churn en Telecomunicaciones
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) diseñado para predecir la probabilidad de que un cliente abandone los servicios de una empresa de telecomunicaciones (Churn). La arquitectura es modular, reproducible y está lista para colaboración o despliegue en producción.

---

## 📌 ¿Qué hace este proyecto?
El objetivo principal es entrenar y consumir un modelo predictivo capaz de identificar qué clientes tienen una alta probabilidad de darse de baja. El pipeline automatiza la ingesta de datos, preprocesamiento (limpieza e imputación), el entrenamiento de un modelo de clasificación (RandomForest/LogisticRegression) y su guardado. Además, el proyecto incluye pruebas automatizadas (QA) y una API para consumir el modelo en tiempo real.

---

## 👥 Roles y Responsabilidades
Este proyecto fue desarrollado bajo un entorno simulado de trabajo colaborativo, donde cada miembro del equipo asumió un rol específico con responsabilidades y contratos de interfaz claros:

- **Data Engineer:** Responsable de `src/data_loader.py`. Encargado de la ingesta de datos, limpieza de valores nulos (ej. `TotalCharges`), eliminación de columnas innecesarias, codificación de variables categóricas y particionamiento del dataset.
- **ML Engineer:** Responsable de `src/trainer_model.py`. Encargado de implementar la fábrica de modelos, entrenar el algoritmo seleccionado desde configuración, evaluar métricas (Accuracy, Recall, F1) y serializar el modelo en `models/`.
- **MLOps Engineer:** Responsable de `src/main.py` y `config/params.yaml`. Orquestador del pipeline, encargado de conectar la salida del Data Engineer con la entrada del ML Engineer y garantizar que el pipeline se ejecute sin errores.
- **QA & Production Engineer:** Responsable de `src/predict.py`, `src/api.py` y `test/test_pipeline.py`. Encargado de escribir pruebas unitarias robustas para garantizar la calidad del código y de implementar el script y API para predicciones.

---

## 📊 ¿Qué dataset usa y para qué sirve?
Se utiliza el dataset público **Telco Customer Churn**.
- **Utilidad:** Permite predecir el comportamiento futuro de los clientes a partir de datos históricos, lo cual resulta invaluable para que los equipos de retención ofrezcan promociones proactivas a usuarios en riesgo.
- (Para más detalles, consultar [DATASET.md](DATASET.md)).

---

## 📂 Estructura de Carpetas

```text
churn-mlops-project/
├── config/
│   └── params.yaml          # Configuración centralizada
├── data/
│   ├── raw/                 # WA_Fn-UseC_-Telco-Customer-Churn.csv (NO SUBIR)
│   └── processed/           # (Opcional) Datos limpios
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Rol: Data Engineer
│   ├── model_trainer.py     # Rol: ML Engineer
│   ├── main.py              # Rol: MLOps Engineer
│   └── predict.py           # Rol: QA Engineer
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Rol: QA Engineer
├── models/                  # Modelos .pkl generados (NO SUBIR o subir solo el final)
├── requirements.txt         # Dependencias
├── .gitignore               # Reglas de exclusión
└── README.md                # Este archivo
```

---

## ⚙️ ¿Cómo lo instalo?

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/RobbeAlex/project-root.git
   cd churn-mlops-project
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Añadir el Dataset:**
   Descarga el archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` desde Kaggle y colócalo dentro de la carpeta `data/raw/`.

---

## 🚀 ¿Cómo lo ejecuto?

## 🏠 Opcion A: "Tradicional" (Entorno Virtual)

### 1. Entrenar el Modelo (Pipeline MLOps)
Ejecuta el script principal para procesar los datos, entrenar y guardar el modelo en la carpeta `models/`:
```bash
python -m src.main
```

### 2. Probar Predicción Local (QA)
Para validar que el modelo guardado funciona, puedes ejecutar el script de predicción que evalúa a un cliente ficticio:
```bash
python -m src.predict
```

### 3. Ejecutar Pruebas (Test Pipeline)
Verifica la integridad del pipeline usando `pytest`:
```bash
pytest test/
```

### 4. Lanzar la API (Opcional)
Inicia el servidor local con FastAPI:
```bash
uvicorn src.api:app --reload
```
Una vez iniciado, puedes hacer un POST a `http://127.0.0.1:8000/predict` con los datos del cliente en JSON.

## 🐳 Opcion B: "Dockerizada" 

Para garantizar que el entorno sea idéntico en cualquier computadora, puedes usar **Docker**. Asegúrate de tener el dataset en `data/raw/` antes de construir la imagen.

1. **Construir la Imagen de Docker:**
   ```bash
   docker build -t churn-mlops-app .
   ```

2. **Entrenar el Modelo usando Docker:**
   Si deseas correr el pipeline de entrenamiento completo de forma aislada:
   ```bash
   docker run --rm churn-mlops-app python -m src.main
   ```

3. **Ejecutar Pruebas (Test Pipeline) con Docker:**
   ```bash
   docker run --rm churn-mlops-app pytest test/
   ```

4. **Lanzar la API usando Docker:**
   Levanta el contenedor exponiendo el puerto 8000:
   ```bash
   docker run -p 8000:8000 churn-mlops-app
   ```
   *La API estará disponible en `http://localhost:8000` y puedes realizar predicciones con `POST /predict`.*

---

## 🏆 Resultados del Modelo
*(Al ejecutar el pipeline base con LogisticRegression, se obtienen las siguientes métricas aproximadas)*
- **Accuracy (Precisión Global):** ~ 81.97%
- **Recall (Sensibilidad - Churn):** ~ 59.52%
- **F1-Score:** ~ 63.61%

> *Nota: Estos valores variarán según el `random_state` y los hiperparámetros elegidos en `config/params.yaml`.*

---

## Ejemplos rápido de Inferencia (Uso)
Con la API levantada, puedes enviar una petición HTTP para predecir si un cliente hará Churn o no.

**Petición (`curl`):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure": 12,
           "MonthlyCharges": 95.50,
           "TotalCharges": 1146.0,
           "InternetService_Fiber optic": 1,
           "Contract_Month-to-month": 1,
           "PaymentMethod_Electronic check": 1
         }'
```

**Petición (`json`):**
```json
{
  "tenure": 12,
  "MonthlyCharges": 95.50,
  "TotalCharges": 1146.0,
  "InternetService_Fiber optic": 1,
  "Contract_Month-to-month": 1,
  "PaymentMethod_Electronic check": 1,
  "PaperlessBilling_Yes": 1
}
```

**Salida esperada:**
```json
{
  "churn_prediction": 1,
  "status": "Churn",
  "probability_churn": 0.73,
  "probability_no_churn": 0.27
}
```

---

## 🤖 Contribución de LLM (Inteligencia Artificial)

Se utilizaron modelos de lenguaje grandes (LLMs) como asistencia técnica para agilizar el desarrollo y la depuración del código:

- **Data Engineer:** Utilizó **Gemini** para optimizar la lógica de preprocesamiento, como la limpieza de espacios en blanco y la conversión a numérico en la columna `TotalCharges`.
- **ML Engineer:** Utilizó **Gemini** para estructurar la extracción del cálculo de métricas de desempeño y para revisar el proceso de guardado de los modelos utilizando `joblib`.
- **MLOps Engineer:** Utilizó **ChatGPT / Gemini** para la resolución de conflictos relacionados con rutas de archivos absolutas vs relativas (`os.path`) y para generar plantillas estructurales para `dvc.yaml`.
- **QA & Production Engineer:** Utilizó **Gemini** para bosquejar la estructura inicial de `pytest` (asserts y fixtures) y sugerir la lógica de captura de excepciones en FastAPI para la respuesta de la red en `predict.py`.
