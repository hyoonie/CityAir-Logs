from django.db import migrations

# Niveles anteriores -> niveles del mockup. 'primaria' no tiene equivalente: se conserva tal cual
# y el formulario de edición pide elegir un nivel válido la próxima vez que se edite la escuela.
EQUIVALENCIAS = {
    'preparatoria': 'media_superior',
    'universidad': 'superior',
}


def migrar_niveles(apps, schema_editor):
    Escuela = apps.get_model('core', 'Escuela')
    for anterior, nuevo in EQUIVALENCIAS.items():
        Escuela.objects.filter(nivel=anterior).update(nivel=nuevo)


def revertir_niveles(apps, schema_editor):
    Escuela = apps.get_model('core', 'Escuela')
    for anterior, nuevo in EQUIVALENCIAS.items():
        Escuela.objects.filter(nivel=nuevo).update(nivel=anterior)


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_cargar_catalogo_geo'),
    ]

    operations = [
        migrations.RunPython(migrar_niveles, revertir_niveles),
    ]
