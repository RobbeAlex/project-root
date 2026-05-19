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
Este proyecto fue desarrollado bajo un entorno simulado de trabajo colaborativo, con el propósito de representar la distribución de funciones que normalmente se presenta en un equipo orientado a la implementación de soluciones de Machine Learning y MLOps. En este contexto, cada integrante asumió un rol específico con responsabilidades delimitadas y una clara relación con los componentes del sistema:

* **Data Engineer:** Responsable de `src/data_loader.py`. Se encargó de la ingesta de datos, la depuración de valores nulos, la eliminación de atributos innecesarios, la codificación de variables categóricas y la partición del conjunto de datos para su posterior procesamiento.
* **ML Engineer:** Responsable de `src/trainer_model.py`. Su función consistió en implementar la lógica de entrenamiento, seleccionar el algoritmo definido en la configuración, evaluar el desempeño mediante métricas como Accuracy, Recall y F1, y serializar el modelo final en el directorio `models/`.
* **MLOps Engineer:** Responsable de `src/main.py` y `config/params.yaml`. Se ocupó de la orquestación del pipeline, la administración de parámetros de ejecución y la integración entre las distintas etapas del flujo de trabajo, con el fin de asegurar su reproducibilidad y correcta ejecución.
* **QA & Production Engineer:** Responsable de `src/predict.py`, `src/api.py` y `test/test_pipeline.py`. Su labor estuvo orientada a la validación funcional mediante pruebas automatizadas, así como al desarrollo de la interfaz de inferencia para el consumo del modelo en un entorno de producción.

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
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # (Opcional) Datos limpios
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Rol: Data Engineer
│   ├── trainer_model.py     # Rol: ML Engineer
│   ├── main.py              # Rol: MLOps Engineer
│   └── predict.py           # Rol: QA Engineer
├── test/
│   ├── __init__.py
│   └── test_pipeline.py     # Rol: QA Engineer
├── models/                  # Modelos .pkl generados (Ignorados por git, generar localmente)
├── requirements.txt         # Dependencias
├── .gitignore               # Reglas de exclusión
├── .dockerignore            # Reglas de exclusión de Docker
├── Dockerfile               # Lista secuencial de comandos e instrucciones para construir una imagen de Docker
├── DATASET.md               # Información del dataset
├── ETHICS.md                # Define los principios éticos
└── README.md                # Este archivo
```

---

## ⚙️ Configuración Inicial (Credenciales de Kaggle)
Para garantizar la reproducibilidad y permitir que el sistema descargue los datos de forma automática, debes configurar tus credenciales de Kaggle:

* Ve a tu perfil de Kaggle -> Settings -> API y haz clic en Create New Token para descargar tu archivo kaggle.json.

* En la raíz del proyecto, crea un archivo llamado .env (este archivo está protegido por .gitignore).

* Abre tu kaggle.json, copia tus datos y colócalos en el archivo .env con el siguiente formato:

```bash
KAGGLE_USERNAME=tu_usuario_de_kaggle
KAGGLE_KEY=tu_token_largo_alfanumerico
```

---

## 🚀 ¿Cómo lo ejecuto?

### 🏠 Opción A: "Tradicional" (Entorno Virtual)

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

4. **Entrenar el Modelo (Pipeline MLOps):**
   El script detectará que no tienes los datos y los descargará automáticamente usando las credenciales del .env:

```bash
python -m src.main
```

5. **Probar Predicción Local y Tests:**

```bash
python -m src.predict
pytest test/
```

6. **Lanzar la API:**

```bash
uvicorn src.api:app --reload
```
Una vez iniciado, puedes hacer un POST a `http://127.0.0.1:8000/predict` con los datos del cliente en JSON.

---

### 🐳 Opción B: "Dockerizada" (Aislamiento Total)

Docker permite correr todo el ecosistema sin instalar Python en tu máquina local. Al usar --env-file .env, compartimos de forma segura las credenciales de Kaggle con el contenedor en tiempo de ejecución.

1. **Construir la Imagen de Docker:**
   ```bash
   docker build -t churn-mlops-app .
   ```

2. **Entrenar el Modelo usando Docker:**
   ```bash
  docker run --rm --env-file .env -v "${PWD}/models:/app/models" -v "${PWD}/data:/app/data" churn-mlops-app python -m src.main
   ```

3. **Ejecutar Pruebas (Test Pipeline) con Docker:**
   ```bash
   docker run --rm --env-file .env churn-mlops-app pytest test/
   ```

4. **Lanzar la API usando Docker:**
   ```bash
   docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/models:/app/models" churn-mlops-app
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

## Ejemplos rápidos de inferencia (Uso)
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
