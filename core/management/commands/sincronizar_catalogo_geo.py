from django.core.management.base import BaseCommand, CommandError

from core import catalogo_geo
from core.models import SincronizacionCatalogo


class Command(BaseCommand):
    help = (
        'Actualiza el catálogo de estados y municipios de México desde el servicio web de INEGI. '
        'Pensado para ejecutarse cada semana (cron). No borra nada: las bajas quedan como inactivas.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Muestra los cambios sin guardarlos.')
        parser.add_argument('--archivo', help='Usa un JSON local (mismo formato de core/data) en lugar de la API.')

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        try:
            datos = catalogo_geo.cargar_archivo(options['archivo']) if options['archivo'] else catalogo_geo.descargar_catalogo()
            resumen = catalogo_geo.aplicar_catalogo(datos, dry_run=dry_run)
        except Exception as e:
            if not dry_run:
                SincronizacionCatalogo.objects.create(exitosa=False, error=str(e))
            # Salida distinta de 0 para que el cron/monitoreo detecte el fallo
            raise CommandError(f'Sincronización fallida: {e}')

        if not dry_run:
            SincronizacionCatalogo.objects.create(exitosa=True, resumen=resumen)

        self.stdout.write(self.style.SUCCESS(f"{'[dry-run] ' if dry_run else ''}Catálogo revisado."))
        for clave, valor in resumen.items():
            if clave != 'detalle':
                self.stdout.write(f'  {clave}: {valor}')
        for linea in resumen['detalle'][:50]:
            self.stdout.write(f'  - {linea}')
        if len(resumen['detalle']) > 50:
            self.stdout.write(f"  ... y {len(resumen['detalle']) - 50} cambios más")
