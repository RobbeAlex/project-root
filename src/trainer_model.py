import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train_and_save_model(X_train, y_train, X_test, y_test, config):
    """
    Entrena un modelo de clasificación, calcula sus métricas y lo guarda.
    """
    model_name = config['model']['name']
    
    # 1. Fábrica de Modelos
    if model_name == "RandomForest":
        n_estimators = config['model'].get('n_estimators', 100)
        max_depth = config['model'].get('max_depth', None)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=config['data']['random_state']
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=1000, 
            random_state=config['data']['random_state']
        )
    else:
        raise ValueError(f"Modelo {model_name} no soportado.")
        
    # 2. Entrenar el modelo
    model.fit(X_train, y_train)
    
    # 3. Calcular las 3 métricas
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print(f"Métricas del modelo {model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 4. Guardar el modelo entrenado con joblib
    model_save_path = config['paths']['model_save']
    joblib.dump(model, model_save_path)
    print(f"Modelo guardado exitosamente en: {model_save_path}")
    
    return metrics