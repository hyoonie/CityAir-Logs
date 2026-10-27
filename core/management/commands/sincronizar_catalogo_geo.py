from django.core.management.base import BaseCommand, CommandError

from core.catalogo_geo import (
    CatalogoInvalido,
    aplicar_catalogo,
    cargar_archivo,
    descargar_catalogo,
    exportar_archivo,
)


class Command(BaseCommand):
    help = 'Sincroniza el catálogo de estados y municipios de México con el servicio web de INEGI'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Muestra los cambios sin guardarlos')
        parser.add_argument('--desde-archivo', metavar='RUTA', nargs='?', const='',
                            help='Usa un JSON local en lugar de INEGI (sin ruta: la semilla del repo)')
        parser.add_argument('--exportar', metavar='RUTA',
                            help='Solo descarga de INEGI y guarda el JSON en RUTA, sin tocar la BD')

    def handle(self, *args, **options):
        try:
            if options['desde_archivo'] is not None:
                datos = cargar_archivo(options['desde_archivo'] or None)
            else:
                self.stdout.write('Descargando catálogo de INEGI...')
                datos = descargar_catalogo()

            if options['exportar']:
                exportar_archivo(datos, options['exportar'])
                self.stdout.write(self.style.SUCCESS(
                    f"Exportados {len(datos['estados'])} estados y {len(datos['municipios'])} municipios "
                    f"a {options['exportar']}"
                ))
                return

            resumen = aplicar_catalogo(datos, dry_run=options['dry_run'])
        except CatalogoInvalido as e:
            raise CommandError(str(e))

        for linea in resumen.pop('detalle'):
            self.stdout.write(f'  {linea}')
        for clave, valor in resumen.items():
            self.stdout.write(f'{clave}: {valor}')
        if options['dry_run']:
            self.stdout.write(self.style.WARNING('Dry run: no se guardó ningún cambio.'))
        else:
            self.stdout.write(self.style.SUCCESS('Catálogo sincronizado.'))
