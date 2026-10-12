import json
import re
import unicodedata
from pathlib import Path

from django.db import migrations
from django.db.models import Q

ARCHIVO = Path(__file__).resolve().parent.parent / 'data' / 'catalogo_geo_mx.json'

# Nombres cortos de uso común -> cve_ent. 'México' a secas es ambiguo (país o estado): no se empareja.
ALIAS_ESTADOS = {
    'veracruz': '30', 'coahuila': '05', 'michoacan': '16', 'cdmx': '09', 'df': '09',
    'distrito federal': '09', 'estado de mexico': '15', 'edo de mexico': '15', 'edomex': '15',
}

# Prefijos de cargos escritos a mano -> código del catálogo cerrado
PREFIJOS_CARGOS = (
    ('subdir', 'subdirector'), ('sudir', 'subdirector'),
    ('dir', 'director'),
    ('coord', 'coordinador'),
)


def normalizar(texto):
    texto = unicodedata.normalize('NFKD', texto or '').encode('ascii', 'ignore').decode('ascii').lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', texto)).strip()


def cargar_catalogo(apps, schema_editor):
    Estado = apps.get_model('core', 'Estado')
    Municipio = apps.get_model('core', 'Municipio')
    Escuela = apps.get_model('core', 'Escuela')
    Usuario = apps.get_model('core', 'Usuario')

    with open(ARCHIVO, encoding='utf-8') as f:
        datos = json.load(f)

    Estado.objects.bulk_create([Estado(**e) for e in datos['estados']], ignore_conflicts=True)
    estados = {e.cve_ent: e for e in Estado.objects.all()}
    Municipio.objects.bulk_create([
        Municipio(cvegeo=m['cvegeo'], estado=estados[m['cve_ent']], cve_mun=m['cve_mun'], nombre=m['nombre'])
        for m in datos['municipios']
    ], ignore_conflicts=True)

    # ── Escuelas existentes: se emparejan por nombre sin acentos (+ alias); lo demás queda para revisión manual
    por_nombre_estado = {normalizar(e.nombre): e for e in estados.values()}
    por_nombre_estado.update({alias: estados[cve] for alias, cve in ALIAS_ESTADOS.items()})
    for escuela in Escuela.objects.filter(Q(pais='mexico') | Q(pais__isnull=True)):
        estado = por_nombre_estado.get(normalizar(escuela.estado))
        if estado is None:
            continue
        texto = normalizar(escuela.municipio)
        candidatos = {texto, normalizar(re.split(r'\s*[-–]\s*', escuela.municipio)[0])}  # 'Xalapa-Enriquez' -> 'xalapa'
        municipios = Municipio.objects.filter(estado=estado)
        municipio = next((m for m in municipios if normalizar(m.nombre) in candidatos), None)
        if municipio is None:
            continue
        escuela.estado_catalogo = estado
        escuela.municipio_catalogo = municipio
        escuela.estado = estado.nombre
        escuela.municipio = municipio.nombre
        if escuela.pais is None:
            escuela.pais = 'mexico'
        escuela.save()

    # ── Cargos escritos a mano -> catálogo cerrado; lo que no coincida se vuelve a capturar al editar
    for usuario in Usuario.objects.exclude(cargo__isnull=True).exclude(cargo=''):
        texto = normalizar(usuario.cargo)
        nuevo = next((codigo for prefijo, codigo in PREFIJOS_CARGOS if texto.startswith(prefijo)), None)
        if usuario.cargo != nuevo:
            usuario.cargo = nuevo
            usuario.save(update_fields=['cargo'])


def deshacer(apps, schema_editor):
    # Los textos de estado/municipio se conservan; solo se vacían las tablas y las referencias
    apps.get_model('core', 'Escuela').objects.update(estado_catalogo=None, municipio_catalogo=None)
    apps.get_model('core', 'Municipio').objects.all().delete()
    apps.get_model('core', 'Estado').objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_catalogo_geo_y_cargos'),
    ]

    operations = [
        migrations.RunPython(cargar_catalogo, deshacer),
    ]
