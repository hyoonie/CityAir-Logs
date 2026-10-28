"""Catálogo de estados y municipios de México (INEGI, servicio web wscatgeo).

La app nunca consulta a INEGI durante una petición del usuario: lee de las tablas locales.
Este módulo descarga el catálogo y lo aplica a la BD (lo usa `sincronizar_catalogo_geo`)
y la migración 0008 carga la semilla de `core/data/catalogo_geo_mx.json`.
"""
import json
import time
from pathlib import Path

import requests
from django.db import transaction

URL_BASE = 'https://gaia.inegi.org.mx/wscatgeo/v2'
ARCHIVO_SEMILLA = Path(__file__).resolve().parent / 'data' / 'catalogo_geo_mx.json'

# Una respuesta con menos filas se considera incompleta y no se aplica (hoy: 32 y ~2,478)
MINIMO_ESTADOS = 32
MINIMO_MUNICIPIOS = 2400
TIMEOUT = 30
REINTENTOS = 3


class CatalogoInvalido(Exception):
    pass


def _get_datos(sesion, url):
    ultimo_error = None
    for intento in range(1, REINTENTOS + 1):
        try:
            respuesta = sesion.get(url, timeout=TIMEOUT)
            respuesta.raise_for_status()
            return respuesta.json()['datos']
        except (requests.RequestException, ValueError, KeyError) as e:
            ultimo_error = e
            time.sleep(intento)
    raise CatalogoInvalido(f'No se pudo consultar {url}: {ultimo_error}')


def descargar_catalogo(sesion=None):
    """Descarga estados y municipios desde INEGI (no requiere token)."""
    sesion = sesion or requests.Session()
    sesion.headers.setdefault('User-Agent', 'CityAir-Logs')
    estados = [
        {'cve_ent': e['cve_ent'], 'nombre': e['nomgeo']}
        for e in _get_datos(sesion, f'{URL_BASE}/mgee/')
    ]
    municipios = []
    for estado in estados:
        for m in _get_datos(sesion, f"{URL_BASE}/mgem/{estado['cve_ent']}"):
            municipios.append({
                'cvegeo': m['cvegeo'], 'cve_ent': m['cve_ent'],
                'cve_mun': m['cve_mun'], 'nombre': m['nomgeo'],
            })
    return {'estados': estados, 'municipios': municipios}


def cargar_archivo(ruta=None):
    with open(ruta or ARCHIVO_SEMILLA, encoding='utf-8') as f:
        datos = json.load(f)
    return {'estados': datos['estados'], 'municipios': datos['municipios']}


def exportar_archivo(datos, ruta):
    """Un registro por línea para que los diffs entre sincronizaciones sean legibles."""
    def lista(registros):
        return ',\n'.join('  ' + json.dumps(r, ensure_ascii=False) for r in registros)

    with open(ruta, 'w', encoding='utf-8') as f:
        f.write('{\n"estados": [\n' + lista(datos['estados']) + '\n],\n')
        f.write('"municipios": [\n' + lista(datos['municipios']) + '\n]\n}\n')


def validar_catalogo(datos, minimo_estados=MINIMO_ESTADOS, minimo_municipios=MINIMO_MUNICIPIOS):
    """Evita aplicar una respuesta incompleta o con claves inconsistentes."""
    estados, municipios = datos.get('estados', []), datos.get('municipios', [])
    if len(estados) < minimo_estados:
        raise CatalogoInvalido(f'Respuesta incompleta: {len(estados)} estados (mínimo {minimo_estados}).')
    if len(municipios) < minimo_municipios:
        raise CatalogoInvalido(f'Respuesta incompleta: {len(municipios)} municipios (mínimo {minimo_municipios}).')
    claves = {e['cve_ent'] for e in estados}
    if len(claves) != len(estados):
        raise CatalogoInvalido('Hay claves de estado repetidas.')
    if len({m['cvegeo'] for m in municipios}) != len(municipios):
        raise CatalogoInvalido('Hay claves de municipio repetidas.')
    huerfanos = [m['cvegeo'] for m in municipios if m['cve_ent'] not in claves]
    if huerfanos:
        raise CatalogoInvalido(f'Municipios sin estado en el catálogo: {huerfanos[:5]}')


def aplicar_catalogo(datos, dry_run=False, Estado=None, Municipio=None, Escuela=None, **minimos):
    """Aplica el catálogo por clave (upsert). No borra nada: las bajas pasan a activo=False.

    Los modelos se pueden pasar como argumento para usar esta función desde una migración
    (modelos históricos). Devuelve un resumen de cambios; con dry_run no se guarda nada.
    """
    if Estado is None:
        from .models import Escuela, Estado, Municipio

    validar_catalogo(datos, **minimos)
    resumen = {
        'estados_nuevos': 0, 'estados_renombrados': 0, 'estados_baja': 0, 'estados_reactivados': 0,
        'municipios_nuevos': 0, 'municipios_renombrados': 0, 'municipios_baja': 0, 'municipios_reactivados': 0,
        'detalle': [],
    }

    with transaction.atomic():
        estados_bd = {e.cve_ent: e for e in Estado.objects.all()}
        claves_estados = set()
        for e in datos['estados']:
            claves_estados.add(e['cve_ent'])
            actual = estados_bd.get(e['cve_ent'])
            if actual is None:
                estados_bd[e['cve_ent']] = Estado.objects.create(cve_ent=e['cve_ent'], nombre=e['nombre'])
                resumen['estados_nuevos'] += 1
                resumen['detalle'].append(f"Estado nuevo: {e['cve_ent']} {e['nombre']}")
                continue
            cambios = []
            if actual.nombre != e['nombre']:
                resumen['estados_renombrados'] += 1
                resumen['detalle'].append(f"Estado renombrado: {e['cve_ent']} '{actual.nombre}' -> '{e['nombre']}'")
                actual.nombre = e['nombre']
                cambios.append('nombre')
            if not actual.activo:
                actual.activo = True
                resumen['estados_reactivados'] += 1
                cambios.append('activo')
            if cambios:
                actual.save(update_fields=cambios)

        for clave, estado in estados_bd.items():
            if clave not in claves_estados and estado.activo:
                estado.activo = False
                estado.save(update_fields=['activo'])
                resumen['estados_baja'] += 1
                resumen['detalle'].append(f'Estado dado de baja: {clave} {estado.nombre}')

        municipios_bd = {m.cvegeo: m for m in Municipio.objects.all()}
        claves_municipios = set()
        nuevos, modificados = [], []
        for m in datos['municipios']:
            claves_municipios.add(m['cvegeo'])
            actual = municipios_bd.get(m['cvegeo'])
            if actual is None:
                nuevos.append(Municipio(
                    cvegeo=m['cvegeo'], estado=estados_bd[m['cve_ent']], cve_mun=m['cve_mun'], nombre=m['nombre'],
                ))
                continue
            modificado = False
            if actual.nombre != m['nombre']:
                resumen['municipios_renombrados'] += 1
                resumen['detalle'].append(f"Municipio renombrado: {m['cvegeo']} '{actual.nombre}' -> '{m['nombre']}'")
                actual.nombre = m['nombre']
                modificado = True
            if not actual.activo:
                actual.activo = True
                resumen['municipios_reactivados'] += 1
                modificado = True
            if modificado:
                modificados.append(actual)

        Municipio.objects.bulk_create(nuevos, batch_size=500)
        resumen['municipios_nuevos'] = len(nuevos)

        for cvegeo, municipio in municipios_bd.items():
            if cvegeo not in claves_municipios and municipio.activo:
                municipio.activo = False
                modificados.append(municipio)
                resumen['municipios_baja'] += 1
                resumen['detalle'].append(f'Municipio dado de baja: {cvegeo} {municipio.nombre}')
        if modificados:
            Municipio.objects.bulk_update(modificados, ['nombre', 'activo'], batch_size=500)

        _refrescar_copias_de_nombre(Escuela)

        if dry_run:
            transaction.set_rollback(True)

    return resumen


def _refrescar_copias_de_nombre(Escuela):
    """Las escuelas guardan una copia de texto del nombre: se actualiza si el catálogo lo renombró."""
    escuelas = Escuela.objects.filter(municipio_catalogo__isnull=False).select_related(
        'estado_catalogo', 'municipio_catalogo')
    for escuela in escuelas:
        estado = escuela.estado_catalogo.nombre if escuela.estado_catalogo else escuela.estado
        municipio = escuela.municipio_catalogo.nombre
        if escuela.estado != estado or escuela.municipio != municipio:
            Escuela.objects.filter(pk=escuela.pk).update(estado=estado, municipio=municipio)
