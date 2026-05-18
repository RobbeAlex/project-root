from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml
import os

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para predecir si un cliente abandonará el servicio (Churn)",
    version="1.0.0"
)

# Definir la estructura de entrada esperada
# Como no conocemos de antemano todas las columnas dummy de un cliente nuevo, 
# permitiremos un JSON dinámico
class CustomerData(BaseModel):
    data: dict

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    root = get_project_root()
    full_path = os.path.join(root, "config/params.yaml")
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

# Cargar el modelo globalmente
config = load_config()
root = get_project_root()
model_path = os.path.join(root, config['paths']['model_save'])
model = None

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("Advertencia: El modelo no existe en la ruta al iniciar la API.")

@app.get("/")
def home():
    return {"message": "Bienvenido a la API de Predicción de Churn. Usa POST /predict"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Entrena el modelo primero.")
    
    if not hasattr(model, 'feature_names_in_'):
        raise HTTPException(status_code=500, detail="El modelo no guarda 'feature_names_in_'. No se puede mapear el input.")
        
    features = model.feature_names_in_
    
    # Crear dataframe base con ceros
    df = pd.DataFrame(0, index=[0], columns=features)
    
    # Rellenar con los datos enviados por el usuario
    for key, value in customer.data.items():
        if key in features:
            df.at[0, key] = value
            
    # Predicción
    prediction = model.predict(df)[0]
    
    result = {
        "churn_prediction": int(prediction),
        "status": "Churn" if prediction == 1 else "No Churn"
    }
    
    # Si soporta probabilidad
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df)[0]
        result["probability_churn"] = float(prob[1])
        result["probability_no_churn"] = float(prob[0])
        
    return result

# Ejemplo de uso con curl a incluir en README:
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": {"tenure": 12, "MonthlyCharges": 50.0, "TotalCharges": 600.0, "InternetService_Fiber optic": 1}}'
