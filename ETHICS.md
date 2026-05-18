# Ética, Transparencia y Sesgos del Modelo (MLOps)

El uso de Machine Learning para predecir el Churn tiene impactos directos en cómo una empresa trata a sus clientes. Es esencial garantizar que el modelo no perpetúe prácticas discriminatorias ni utilice datos de manera no ética.

## ⚠️ Sesgos y Limitaciones del Modelo y Dataset

### 1. Grupos Subrepresentados
- **Adultos Mayores (Senior Citizens):** El dataset contiene un desbalance significativo entre clientes jóvenes/adultos versus adultos mayores. El modelo podría no haber aprendido suficientes patrones precisos sobre este grupo, provocando falsos positivos (asumir que abandonarán cuando no es cierto) o falsos negativos.
- **Clientes sin acceso a Internet:** El volumen de datos puede inclinarse fuertemente hacia los clientes que consumen fibra óptica, marginando a aquellos que solo usan servicio telefónico.

### 2. Prejuicios Históricos en las Etiquetas
- Las etiquetas de `Churn` reflejan las bajas históricas. Si históricamente el servicio de atención al cliente de la empresa fue deficiente en ciertas zonas geográficas o para clientes con contratos más baratos, el modelo aprenderá a penalizar esos atributos, asumiendo que el "problema" está en el tipo de cliente y no en la calidad del servicio ofrecido.

### 3. Variables Sensibles Identificadas
El dataset incluye atributos que requieren especial atención ética:
- **`gender` (Género):** Utilizar el género para predecir si alguien cancelará su contrato puede rozar la discriminación comercial. Ofrecer mejores promociones o atención prioritaria a un género por sobre otro basado en las salidas del algoritmo es una práctica no ética.
- **`SeniorCitizen` (Edad/Adulto Mayor):** Penalar a los adultos mayores o perfilarlos automáticamente puede violar regulaciones de equidad al consumidor.
- **`Partner` y `Dependents` (Estado civil / Familiar):** Asumir el comportamiento de un individuo con base en su estructura familiar.
- **`PaymentMethod` (Método de Pago) / `MonthlyCharges` (Cargos):** Al carecer de datos explícitos sobre ingresos, estas variables actúan como *proxys* (sustitutos) de la clase social. Clientes que pagan con cheque electrónico o mes a mes suelen tener menor liquidez; el modelo podría discriminar a personas de bajos ingresos.

## 🛡️ Mitigación y Transparencia
- **Monitoreo Continuo:** Debe implementarse un monitoreo de *Fairness* (Equidad) durante el ciclo de vida de MLOps para asegurar que las tasas de error (Falsos Positivos) se distribuyan equitativamente entre géneros y edades.
- **Exclusión de Features:** Para futuras versiones del modelo en producción, se recomienda realizar un experimento eliminando la variable `gender` y `SeniorCitizen` para verificar si la caída en el Accuracy es asumible a cambio de un modelo ciego a características protegidas.
