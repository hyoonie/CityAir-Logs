# 1. IMAGEN BASE
# Usamos Python 3.12 en su versión "slim" (ligera, basada en Debian Linux)
FROM python:3.12-slim

# 2. VARIABLES DE ENTORNO
# Evita que Python genere archivos .pyc (basura en contenedores)
ENV PYTHONDONTWRITEBYTECODE=1
# Hace que los logs de Python (print) se vean de inmediato en la consola
ENV PYTHONUNBUFFERED=1

# 3. DIRECTORIO DE TRABAJO
# Crea la carpeta /app dentro del contenedor y se mete ahí
WORKDIR /app

# 4. DEPENDENCIAS DEL SISTEMA (LINUX)
# Instalamos herramientas necesarias para compilar mysqlclient y otras librerías
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* # Limpieza para que la imagen pese menos

# 5. DEPENDENCIAS DE PYTHON
# Copiamos primero solo el requirements.txt para aprovechar la caché de Docker
COPY requirements.txt /app/
# Instalamos las librerías (Django, PyMongo, Paho-MQTT, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 6. CÓDIGO FUENTE
# Copiamos todo tu proyecto dentro del contenedor
COPY . /app/

# 7. PUERTO
# Informamos que el contenedor escuchará en el 8000
EXPOSE 8000

# 8. COMANDO POR DEFECTO
# Este comando inicia el servidor web (Gunicorn)
# ⚠️ IMPORTANTE: Asegúrate de que 'CityAirLogs' sea el nombre real de la carpeta donde está wsgi.py
CMD ["gunicorn", "CityAirLogs.wsgi:application", "--bind", "0.0.0.0:8000"]