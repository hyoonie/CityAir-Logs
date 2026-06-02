from django.db import migrations


NUEVOS_TIPOS = [
    (4, 'Enlace',  'Encargado de proyecto en la escuela. Puede crear cuentas de Docentes y Alumnos.'),
    (5, 'Docente', 'Docente participante. Accede a datos avanzados y lista de alumnos de su escuela.'),
    (6, 'Alumno',  'Alumno participante. Accede a datos de su escuela y comparativas con otras escuelas.'),
]


def insertar_tipos(apps, schema_editor):
    TipoUsuario = apps.get_model('core', 'TipoUsuario')
    for id_tipo, nombre, descripcion in NUEVOS_TIPOS:
        TipoUsuario.objects.get_or_create(
            id_tipo=id_tipo,
            defaults={'nombre_tipo': nombre, 'descripcion': descripcion}
        )


def eliminar_tipos(apps, schema_editor):
    TipoUsuario = apps.get_model('core', 'TipoUsuario')
    TipoUsuario.objects.filter(id_tipo__in=[4, 5, 6]).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_escuela_model_y_relaciones'),
    ]

    operations = [
        migrations.RunPython(insertar_tipos, eliminar_tipos),
    ]
