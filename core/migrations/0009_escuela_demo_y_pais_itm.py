from decimal import Decimal

from django.db import migrations


# Las cuentas demo de rol escolar viven en una escuela propia para no mezclar datos con escuelas reales
USUARIOS_DEMO_ESCOLARES = ['demo_enlace', 'demo_docente', 'demo_alumno']
ESCUELA_DEMO = 'Escuela Demo CityAir'
ESCUELA_ITM = 'Institución Universitaria ITM'


def asignar_escuela_demo(apps, schema_editor):
    Usuario = apps.get_model('core', 'Usuario')
    Escuela = apps.get_model('core', 'Escuela')
    Estado = apps.get_model('core', 'Estado')
    Municipio = apps.get_model('core', 'Municipio')

    usuarios = Usuario.objects.filter(usuario__in=USUARIOS_DEMO_ESCOLARES)
    # Sin cuentas demo (BD nueva o de pruebas) no se crea la escuela
    if not usuarios.exists():
        return

    estado = Estado.objects.filter(cve_ent='30').first()        # Veracruz
    municipio = Municipio.objects.filter(cvegeo='30087').first()  # Xalapa
    escuela, _ = Escuela.objects.get_or_create(
        nombre=ESCUELA_DEMO,
        defaults={
            'nivel': 'universidad',
            'sostenimiento': 'publico',
            'turno': 'matutino',
            'correo_institucional': 'escuela@demo.cityair.mx',
            'pais': 'mexico',
            'estado_catalogo': estado,
            'municipio_catalogo': municipio,
            'estado': estado.nombre if estado else 'Veracruz de Ignacio de la Llave',
            'municipio': municipio.nombre if municipio else 'Xalapa',
            'latitud': Decimal('19.543775'),
            'longitud': Decimal('-96.910435'),
        },
    )
    usuarios.update(escuela=escuela)


def corregir_pais_itm(apps, schema_editor):
    # La 0007 dejó a todas las escuelas con país México por defecto; el ITM está en Medellín, Colombia
    Escuela = apps.get_model('core', 'Escuela')
    Escuela.objects.filter(nombre=ESCUELA_ITM, pais='mexico').update(
        pais='colombia', estado_catalogo=None, municipio_catalogo=None,
    )


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_cargar_catalogo_geo'),
    ]

    # Correcciones de datos: al revertir no se deshacen
    operations = [
        migrations.RunPython(asignar_escuela_demo, migrations.RunPython.noop),
        migrations.RunPython(corregir_pais_itm, migrations.RunPython.noop),
    ]
