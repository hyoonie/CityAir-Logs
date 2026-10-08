from django.db import migrations


# Catálogo anterior -> catálogo nuevo
MAPEO_NIVELES = {
    'preparatoria': 'media_superior',
    'universidad': 'superior',
}
# Sin reverso: al revertir no se distingue de las secundarias originales
MAPEO_SIN_REVERSO = {
    'primaria': 'secundaria',
}


def migrar_niveles(apps, schema_editor):
    Escuela = apps.get_model('core', 'Escuela')
    for anterior, nuevo in {**MAPEO_NIVELES, **MAPEO_SIN_REVERSO}.items():
        Escuela.objects.filter(nivel=anterior).update(nivel=nuevo)


def revertir_niveles(apps, schema_editor):
    Escuela = apps.get_model('core', 'Escuela')
    for anterior, nuevo in MAPEO_NIVELES.items():
        Escuela.objects.filter(nivel=nuevo).update(nivel=anterior)


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_escuela_datos_ampliados'),
    ]

    operations = [
        migrations.RunPython(migrar_niveles, revertir_niveles),
    ]
