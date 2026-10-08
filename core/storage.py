import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage


def get_convenios_storage():
    """Almacenamiento privado de convenios (PDF): fuera de static y sin URL pública.
    La ruta se toma de CONVENIOS_ROOT para no depender de settings.py (no versionado)."""
    location = os.getenv('CONVENIOS_ROOT') or os.path.join(settings.BASE_DIR, 'private_media', 'convenios')
    return FileSystemStorage(location=location)
