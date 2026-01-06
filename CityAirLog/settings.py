"""
Django settings for CityAirLog project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# =========================
# BASE
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# =========================
# SECURITY
# =========================
SECRET_KEY = 'django-insecure-509bq=ym+_-%4tz-(9_9w5p+hwa*1mv^qm7378)0n*u0$h%x#5'
DEBUG = True
ALLOWED_HOSTS = []

# =========================
# APPLICATIONS
# =========================
INSTALLED_APPS = [
    'rest_framework',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'CityAirLog.urls'

# =========================
# TEMPLATES
# =========================
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],   # templates globales
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'CityAirLog.wsgi.application'

# =========================
# DATABASE
# =========================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# =========================
# PASSWORDS
# =========================
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# =========================
# INTERNATIONALIZATION
# =========================
LANGUAGE_CODE = 'es-mx'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# =========================
# STATIC FILES
# =========================
STATIC_URL = '/static/'

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'core', 'static'),
]

# Para producción (opcional)
# STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# =========================
# DEFAULT PK
# =========================
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================================
# RUTAS DEL PIPELINE DE DATOS / MODELOS (ACTUAL)
# ============================================================

# Carpeta donde se colocan archivos de entrada (si aplica)
INPUT_DIR_BASE = BASE_DIR / 'data_input'

# Carpeta donde se guardan TODOS los resultados del modelo
OUT_DIR = BASE_DIR / 'data_output'

# ============================================================
# CONFIGURACIÓN MONGODB
# ============================================================
MONGO_SETTINGS = {
    "HOST": os.getenv("MONGO_HOST"),
    "PORT": int(os.getenv("MONGO_PORT", 27017)),
    "USERNAME": os.getenv("MONGO_USERNAME"),
    "PASSWORD": os.getenv("MONGO_PASSWORD"),
    "DB_NAME": os.getenv("MONGO_DB_NAME", "CityAirLogs"),
    "AUTH_SOURCE": os.getenv("MONGO_AUTH_SOURCE", "admin"),
    "AUTH_MECHANISM": os.getenv("MONGO_AUTH_MECHANISM", "SCRAM-SHA-1"),
}
