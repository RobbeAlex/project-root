import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config):
    """
    Carga el dataset de Churn, limpia la columna TotalCharges,
    elimina customerID, codifica variables categóricas y
    divide en datos de entrenamiento y prueba.
    """
    # 1. Cargar datos
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, config['paths']['data_raw'])
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR CRÍTICO] El dataset no se encuentra en la ruta: {data_path}")
        print("Para que el proyecto funcione, debes descargar el archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        print("desde Kaggle y colocarlo en la carpeta 'data/raw/'.")
        print("Alternativamente, el repositorio ahora debería incluirlo directamente en Git.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # 2. Limpiar TotalCharges
    # Convertir a numérico (errores='coerce' convertirá los vacíos en NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Imputar los NaN con la mediana
    median_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_charges)
    
    # 3. Eliminar customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # 4. Codificar gender, Partner, Churn a 0/1
    # Se utiliza map para hacer una codificación binaria directa
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    if 'Partner' in df.columns:
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # 5. Convertir el resto de las variables categóricas usando get_dummies
    # Esto es necesario para que modelos como RandomForest o LogisticRegression funcionen
    df = pd.get_dummies(df, drop_first=True)
    
    # Asegurar que todas las columnas son booleanas o numéricas y limpiarlas
    # get_dummies puede generar booleanos, los convertimos a enteros
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
        
    # 6. Dividir usando test_size y random_state de la configuración
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test