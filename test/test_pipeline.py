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
    
    # load_and_preprocess_data se encargará de validar la existencia o descargar el dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    
    # Verificacion de que los datos no esten vacios
    assert not X_train.empty, "X_train está vacío"
    assert not X_test.empty, "X_test está vacío"
    assert not y_train.empty, "y_train está vacío"
    assert not y_test.empty, "y_test está vacío"
    
def test_train_and_save_model():
    config = load_config()
    root = get_project_root()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    
    # Transformar el modelo en uno de menor tamaño para que el test sea más rápido
    if config['model']['name'] == "RandomForest":
        config['model']['n_estimators'] = 5
        
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    
    # Verificación de métricas
    assert isinstance(metrics, dict), "El resultado debe ser un diccionario"
    assert "accuracy" in metrics, "Falta la métrica 'accuracy' en el resultado"
    assert "recall" in metrics, "Falta la métrica 'recall' en el resultado"
    assert "f1_score" in metrics, "Falta la métrica 'f1_score' en el resultado"
    
    # Verificacion de que el modelo haya sido guardado
    model_save_path = os.path.join(root, config['paths']['model_save'])
    assert os.path.exists(model_save_path), "El archivo model.pkl no fue guardado"
