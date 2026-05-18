import joblib
import pandas as pd
import yaml
import os

def load_config(config_path="config/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def predict_example():
    config = load_config()
    model_path = config['paths']['model_save']
    
    if not os.path.exists(model_path):
        print(f"Error: El modelo no se encuentra en la ruta {model_path}. Por favor ejecuta el pipeline de entrenamiento primero (python -m src.main).")
        return
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
        
    print("Modelo cargado exitosamente. Generando predicción para cliente de ejemplo...")
    
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        # Creamos un cliente con valores base (0) para todas las features
        example_data = pd.DataFrame(0, index=[0], columns=features)
        
        # Asignamos valores representativos a algunas columnas clave
        if 'tenure' in features:
            example_data['tenure'] = 24
        if 'MonthlyCharges' in features:
            example_data['MonthlyCharges'] = 85.50
        if 'TotalCharges' in features:
            example_data['TotalCharges'] = 2052.0
        if 'InternetService_Fiber optic' in features:
            example_data['InternetService_Fiber optic'] = 1
        if 'Contract_Month-to-month' in features:
            example_data['Contract_Month-to-month'] = 1
            
        prediction = model.predict(example_data)
        
        # Modelos como LogisticRegression y RandomForest soportan predict_proba
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(example_data)
            prob_str = f"(Probabilidad Churn: {prob[0][1]:.2%})"
        else:
            prob_str = ""
            
        resultado = "VA A ABANDONAR (Churn=1)" if prediction[0] == 1 else "SE QUEDARÁ (Churn=0)"
        print(f"Resultado de predicción para el cliente de ejemplo: {resultado} {prob_str}")
    else:
        print("El modelo no tiene la propiedad 'feature_names_in_'. No se puede automatizar la creación del input.")

if __name__ == "__main__":
    predict_example()