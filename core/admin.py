# core/admin.py

from django.contrib import admin
# Importa todos los modelos que quieres ver
from .models import Usuario, Dispositivo, TipoUsuario, Enfermedad

# Registra cada modelo en el sitio de admin
admin.site.register(Usuario)
admin.site.register(Dispositivo)
admin.site.register(TipoUsuario)
admin.site.register(Enfermedad)
