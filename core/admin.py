# core/admin.py

from django.contrib import admin
# Importa todos los modelos que quieres ver
from .models import Usuario, Dispositivo, TipoUsuario, Enfermedad, Estado, Municipio, SincronizacionCatalogo

# Registra cada modelo en el sitio de admin
admin.site.register(Usuario)
admin.site.register(Dispositivo)
admin.site.register(TipoUsuario)
admin.site.register(Enfermedad)
admin.site.register(Estado)
admin.site.register(Municipio)
admin.site.register(SincronizacionCatalogo)
