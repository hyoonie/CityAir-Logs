import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0009_cargar_catalogo_geo'),
    ]

    operations = [
        # Alumnos sin correo: entran con el usuario autogenerado
        migrations.AlterField(
            model_name='usuario',
            name='email',
            field=models.EmailField(blank=True, max_length=255, null=True, unique=True),
        ),
        migrations.AddField(
            model_name='usuario',
            name='grado',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='usuario',
            name='grupo',
            field=models.CharField(blank=True, max_length=1, null=True),
        ),
        # Primero sin default para que los usuarios existentes queden sin fecha de alta
        # (con default, Django llenaría todas las filas con la fecha de la migración)
        migrations.AddField(
            model_name='usuario',
            name='fecha_alta',
            field=models.DateTimeField(blank=True, editable=False, null=True),
        ),
        migrations.AlterField(
            model_name='usuario',
            name='fecha_alta',
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now, editable=False, null=True),
        ),
        migrations.AddField(
            model_name='documentoescuela',
            name='fecha_vigencia',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='escuela',
            name='aviso_privacidad_aceptado',
            field=models.DateField(blank=True, null=True),
        ),
    ]
