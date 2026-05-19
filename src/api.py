from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
import uvicorn
import pandas as pd
import yaml
import os

app = FastAPI()

# Creacion de la clase para el modelo para soportar los espacios y guiones
class ClienteChurn(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_Fiber_optic: int = Field(alias="InternetService_Fiber optic")
    Contract_Month_to_month: int = Field(alias="Contract_Month-to-month")
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check")
    PaperlessBilling_Yes: int = Field(alias="PaperlessBilling_Yes")

    # Configuración para que pydantic entienda la variable o el alias
    model_config = ConfigDict(populate_by_name=True)

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    root = get_project_root()
    full_path = os.path.join(root, "config/params.yaml")
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

# Carga del modelo al iniciar la API
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
def predict_churn(customer: ClienteChurn):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Entrena el modelo primero.")
    
    if not hasattr(model, 'feature_names_in_'):
        raise HTTPException(status_code=500, detail="El modelo no guarda 'feature_names_in_'. No se puede mapear el input.")
        
    features = model.feature_names_in_
    
    # Extraccion de los datos sin modificar los nombres
    datos_recibidos = customer.model_dump(by_alias=True)

    # Creacion del DataFrame y reindexacion para rellenar con 0 lo que falte sin romper los dtypes
    df_inicial = pd.DataFrame([datos_recibidos])
    
    # Alinear las columnas con las del modelo y rellenar con 0 cualquier variable ausente
    df = df_inicial.reindex(columns=features, fill_value=0)
            
    # Realizar una inferencia con el modelo
    prediction = model.predict(df)[0]
    
    result = {
        "churn_prediction": int(prediction),
        "status": "Churn" if prediction == 1 else "No Churn"
    }
    
    # Se agregan métricas probabilísticas de no churn y churn
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df)[0]
        result["probability_churn"] = float(prob[1])
        result["probability_no_churn"] = float(prob[0])
        
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)