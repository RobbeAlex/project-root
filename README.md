# Proyecto MLOps: Predicción de Churn en Telecomunicaciones

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) diseñado para predecir la probabilidad de que un cliente abandone los servicios de una empresa de telecomunicaciones (Churn). La arquitectura es modular, reproducible y está lista para colaboración o despliegue en producción.

## 📌 ¿Qué hace este proyecto?
El objetivo principal es entrenar y consumir un modelo predictivo capaz de identificar qué clientes tienen una alta probabilidad de darse de baja. El pipeline automatiza la ingesta de datos, preprocesamiento (limpieza e imputación), el entrenamiento de un modelo de clasificación (RandomForest/LogisticRegression) y su guardado. Además, el proyecto incluye pruebas automatizadas (QA) y una API para consumir el modelo en tiempo real.

## 📊 ¿Qué dataset usa y para qué sirve?
Se utiliza el dataset público **Telco Customer Churn**.
- **Utilidad:** Permite predecir el comportamiento futuro de los clientes a partir de datos históricos, lo cual resulta invaluable para que los equipos de retención ofrezcan promociones proactivas a usuarios en riesgo.
- (Para más detalles, consultar [DATASET.md](DATASET.md)).

## ⚙️ ¿Cómo lo instalo?

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_FORK>
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

## 🚀 ¿Cómo lo ejecuto?

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

## 🐳 ¿Cómo lo ejecuto con Docker? (Replicabilidad)

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

## 🏆 Resultados del Modelo
*(Al ejecutar el pipeline base con RandomForest, se obtendrán las siguientes métricas aproximadas)*
- **Accuracy (Precisión Global):** ~ 79.5%
- **Recall (Sensibilidad - Churn):** ~ 51.2%
- **F1-Score:** ~ 58.7%

> *Nota: Estos valores variarán según el `random_state` y los hiperparámetros elegidos en `config/params.yaml`.*

## 🤖 Contribución LLM
- **ChatGPT / Gemini:** Se utilizaron modelos LLM de asistencia para diseñar la arquitectura MLOps base, ayudar en la refactorización modular (creando `data_loader.py` y `trainer_model.py`) y estructurar los archivos Markdown de la documentación (`DATASET.md`, `ETHICS.md`). Además, se empleó para generar la suite de test en `pytest`.
