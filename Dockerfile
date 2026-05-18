# Usamos una imagen ligera de Python 3.11
FROM python:3.11-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Actualizamos paquetes base (opcional pero recomendado)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos primero los requerimientos para aprovechar el caché de Docker
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código fuente al contenedor
COPY . .

# Exponemos el puerto de la API
EXPOSE 8000

# Configuramos la variable de entorno para que Python reconozca la carpeta 'src'
ENV PYTHONPATH=/app

# Comando por defecto al ejecutar el contenedor (Inicia la API)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
