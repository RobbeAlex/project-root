import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config):
    # Cargar datos
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, config['paths']['data_raw'])
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR CRÍTICO] El dataset no se encuentra en la ruta: {data_path}")
        print("Para que el proyecto funcione, debes descargar el archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        print("desde Kaggle y colocarlo en la carpeta 'data/raw/'.")
        print("Alternativamente, el repositorio ahora debería incluirlo directamente en Git.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Limpiar TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    median_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
    
    # Eliminar customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # Codificar variables categóricas binarias
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    if 'Partner' in df.columns:
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # Convertir columnas booleanas a enteros
    df = pd.get_dummies(df, drop_first=True)
    
    # Asegurar que las columnas booleanas sean enteras
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
        
    # Dividir usando test_size y random_state de la configuración
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test