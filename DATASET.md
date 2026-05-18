# Documentación del Dataset: Telco Customer Churn

## 📖 Descripción del Dataset
El dataset **Telco Customer Churn** contiene información sobre una empresa ficticia de telecomunicaciones que proporcionó servicios telefónicos y de internet a 7,043 clientes en California durante el tercer trimestre. Indica qué clientes se han dado de baja, se han quedado o se han suscrito a su servicio.

- **Filas:** 7,043 clientes
- **Columnas:** 21 atributos (features)
- **Tipo de datos:**
  - **Identificador:** `customerID` (alfanumérico, ignorado durante el entrenamiento).
  - **Demográficos:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
  - **Servicios contratados:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
  - **Información de cuenta:** `tenure` (meses en la empresa), `Contract` (tipo de contrato), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` (cargo mensual), `TotalCharges` (cargos totales acumulados).
  - **Target (Variable Objetivo):** `Churn` (Yes / No).

## 🎯 Problema que Resuelve
Este es un problema clásico de **Clasificación Binaria**.
El objetivo es predecir si un cliente abandonará (Churn = Yes = 1) o permanecerá (Churn = No = 0) en base a su perfil, servicios y patrón de pago.

## 💼 Aplicaciones Prácticas
1. **Campañas de Retención Proactivas:** Los equipos de marketing pueden ofrecer descuentos, mejoras de servicio o atención personalizada exclusivamente a los clientes que el modelo clasifique como de "alto riesgo de abandono".
2. **Optimización de Ingresos:** Al identificar de antemano el Churn, la empresa evita los altos costos de adquirir nuevos clientes, rentabilizando su base instalada.
3. **Análisis de Fallas en el Servicio:** Permite encontrar correlaciones (por ejemplo, si todos los clientes con Internet de Fibra Óptica están abandonando, es un indicador de que el servicio puede estar fallando o siendo muy caro comparado a la competencia).

## ⚖️ Implicaciones Éticas y Sesgos Observados
- Al usar variables demográficas (`gender`, `SeniorCitizen`, `Partner`), el modelo podría sesgar sus predicciones y perfilar injustamente a personas de cierta edad o género.
- **Sesgo de exclusión:** No contamos con datos socioeconómicos directos, sin embargo, los métodos de pago o cargos mensuales podrían actuar como variables "proxy" (sustitutas) del nivel de ingresos, llevando al algoritmo a discriminar indirectamente.
*(Para más detalles, ver archivo `ETHICS.md`).*
