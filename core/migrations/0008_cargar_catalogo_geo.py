from django.db import migrations

from core.catalogo_geo import aplicar_catalogo, cargar_archivo


def cargar_catalogo(apps, schema_editor):
    aplicar_catalogo(
        cargar_archivo(),
        Estado=apps.get_model('core', 'Estado'),
        Municipio=apps.get_model('core', 'Municipio'),
        Escuela=apps.get_model('core', 'Escuela'),
    )


def vaciar_catalogo(apps, schema_editor):
    apps.get_model('core', 'Escuela').objects.update(estado_catalogo=None, municipio_catalogo=None)
    apps.get_model('core', 'Municipio').objects.all().delete()
    apps.get_model('core', 'Estado').objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0007_catalogo_geo_y_escuela_ampliada'),
    ]

    operations = [
        migrations.RunPython(cargar_catalogo, vaciar_catalogo),
    ]
