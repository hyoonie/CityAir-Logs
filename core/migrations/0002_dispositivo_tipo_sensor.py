# Migración manual: sincroniza cambios previos no migrados y agrega tipo_sensor
#
# Contexto: la migración inicial (0001) tenía el campo 'coPostal' y no tenía
# 'colonia'. Esos cambios ya se aplicaron directo en MySQL sin migración.
# Esta migración los registra en el historial de Django (sin tocar la BD)
# y agrega el nuevo campo 'tipo_sensor' (este sí es nuevo en la BD).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        # ── Cambios ya existentes en la BD (solo actualiza estado de Django) ──
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RenameField(
                    model_name='dispositivo',
                    old_name='coPostal',
                    new_name='codigo_postal',
                ),
            ],
            database_operations=[],  # Ya está renombrado en la BD
        ),
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.AddField(
                    model_name='dispositivo',
                    name='colonia',
                    field=models.CharField(max_length=128),
                ),
            ],
            database_operations=[],  # Ya existe en la BD
        ),

        # ── Campo nuevo: tipo_sensor (este sí se ejecuta en la BD) ──
        migrations.AddField(
            model_name='dispositivo',
            name='tipo_sensor',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
