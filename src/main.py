import yaml
import os
from src.data_loader import load_and_preprocess_data
from src.trainer_model import train_and_save_model

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path="config/params.yaml"):
    root = get_project_root()
    full_path = os.path.join(root, config_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("Iniciando Pipeline de MLOps para Predicción de Churn...")
    
    # Cargar configuración
    config = load_config()
    print("Configuración cargada correctamente.")
    
    # Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"Datos divididos: {len(X_train)} train, {len(X_test)} test")
    
    # Entrenar y guardar modelo
    print("Entrenando modelo...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    
    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()