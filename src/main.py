import yaml
import os
from src.data_loader import load_and_preprocess_data
from kaggle.api.kaggle_api_extended import KaggleApi
from src.trainer_model import train_and_save_model

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path="config/params.yaml"):
    root = get_project_root()
    full_path = os.path.join(root, config_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def verificar_y_descargar_datos(config):
    root = get_project_root()
    
    # 1. Obtener la ruta absoluta del CSV usando tu configuración existente
    csv_path = os.path.join(root, config['paths']['data_raw'])
    
    # 2. Romper la ruta para saber cuál es la carpeta contenedora (ej. "data/raw")
    raw_data_dir = os.path.dirname(csv_path)
    
    # 3. Obtener el slug de Kaggle
    kaggle_slug = config['data']['kaggle_slug']

    # Si el archivo ya existe, no hace nada
    if os.path.exists(csv_path):
        print("✨ El dataset ya existe localmente. Omitiendo descarga de Kaggle.")
        return

    print(f"📥 Dataset no encontrado en {csv_path}.")
    print(f"Iniciando descarga desde Kaggle: {kaggle_slug}...")
    os.makedirs(raw_data_dir, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
        
        # Descarga y desatasca el zip en la carpeta "data/raw"
        api.dataset_download_files(kaggle_slug, path=raw_data_dir, unzip=True)
        print("✅ Dataset descargado y descomprimido con éxito.")
        
    except Exception as e:
        print(f"❌ Error al conectar con la API de Kaggle: {e}")
        print("Asegúrate de que tu 'kaggle.json' esté en ~/.kaggle/")
        raise e

def main():
    print("Iniciando Pipeline de MLOps para Predicción de Churn...")
    
    # 1. Cargar configuración
    config = load_config()
    print("Configuración cargada correctamente.")
    
    # 2. Asegurar la existencia de los datos (Paso nuevo)
    verificar_y_descargar_datos(config)
    
    # 3. Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"Datos divididos: {len(X_train)} train, {len(X_test)} test")
    
    # 4. Entrenar y guardar modelo
    print("Entrenando modelo...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    
    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()