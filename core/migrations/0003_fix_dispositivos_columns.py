from django.db import migrations, models


def apply_db_changes(apps, schema_editor):
    """Aplica los cambios físicos en la BD solo si son necesarios.
    En producción ya existen, en local todavía no."""
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("""
            SELECT COLUMN_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'dispositivos'
        """)
        existing_cols = [row[0] for row in cursor.fetchall()]

        # Renombrar coPostal → codigo_postal solo si aún existe el nombre viejo
        if 'coPostal' in existing_cols and 'codigo_postal' not in existing_cols:
            cursor.execute(
                "ALTER TABLE dispositivos CHANGE coPostal codigo_postal VARCHAR(8) NOT NULL"
            )

        # Agregar colonia solo si no existe
        if 'colonia' not in existing_cols:
            cursor.execute(
                "ALTER TABLE dispositivos ADD COLUMN colonia VARCHAR(128) NOT NULL DEFAULT ''"
            )


def reverse_db_changes(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_dispositivo_tipo_sensor'),
    ]

    operations = [
        migrations.RunPython(apply_db_changes, reverse_db_changes),
    ]
