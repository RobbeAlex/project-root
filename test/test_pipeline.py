import sys
import os
import yaml
import pytest
from src.trainer_model import train_and_save_model
from src.data_loader import load_and_preprocess_data

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    root = get_project_root()
    full_path = os.path.join(root, "config/params.yaml")
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def test_load_and_preprocess_data():
    config = load_config()
    root = get_project_root()
    
    # Aseguramos que el archivo existe antes de probar
    data_path = os.path.join(root, config['paths']['data_raw'])
    assert os.path.exists(data_path), "El archivo CSV no existe en la ruta especificada."
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    
    # 1. Verificar que load_and_preprocess_data no devuelve datos vacíos
    assert not X_train.empty, "X_train está vacío"
    assert not X_test.empty, "X_test está vacío"
    assert not y_train.empty, "y_train está vacío"
    assert not y_test.empty, "y_test está vacío"
    
def test_train_and_save_model():
    config = load_config()
    root = get_project_root()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    
    # Hacemos el modelo más pequeño y rápido para la prueba si es RandomForest
    if config['model']['name'] == "RandomForest":
        config['model']['n_estimators'] = 5
        
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    
    # 2. Verificar que train_and_save_model devuelve un diccionario con las 3 métricas
    assert isinstance(metrics, dict), "El resultado debe ser un diccionario"
    assert "accuracy" in metrics, "Falta la métrica 'accuracy' en el resultado"
    assert "recall" in metrics, "Falta la métrica 'recall' en el resultado"
    assert "f1_score" in metrics, "Falta la métrica 'f1_score' en el resultado"
    
    # Verificar que el archivo se guardó
    model_save_path = os.path.join(root, config['paths']['model_save'])
    assert os.path.exists(model_save_path), "El archivo model.pkl no fue guardado"
