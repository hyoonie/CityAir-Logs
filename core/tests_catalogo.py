"""Pruebas del catálogo cerrado de estados/municipios (México), cargos y sincronización con INEGI."""
import importlib
from io import StringIO
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.urls import reverse

from core import catalogo_geo
from core.models import Escuela, Estado, Municipio, SincronizacionCatalogo, Usuario
from core.tests import (
    ConveniosTmpMixin, SetUpMixin, datos_enlace, datos_escuela, pdf_falso, veracruz_xalapa,
)


class CatalogoEscuelaTest(ConveniosTmpMixin, SetUpMixin):

    def setUp(self):
        super().setUp()
        self.client.login(username='admin_t', password='Test1234!')
        self.estado, self.municipio = veracruz_xalapa()

    def registrar(self, **extra):
        datos = datos_escuela(**extra)
        datos['archivo'] = pdf_falso()
        return self.client.post(reverse('escuelas-registrar'), datos)

    def test_catalogo_cargado_por_migracion(self):
        self.assertEqual(Estado.objects.count(), 32)
        self.assertGreaterEqual(Municipio.objects.count(), 2400)
        self.assertEqual(self.municipio.nombre, 'Xalapa')
        self.assertEqual(self.estado.nombre, 'Veracruz de Ignacio de la Llave')
        # El nombre oficial más largo supera los 64 caracteres originales
        self.assertGreater(max(len(m.nombre) for m in Municipio.objects.all()), 64)

    def test_mexico_guarda_catalogo_y_copia_del_nombre(self):
        self.registrar()
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        self.assertEqual(escuela.estado_catalogo, self.estado)
        self.assertEqual(escuela.municipio_catalogo, self.municipio)
        self.assertEqual(escuela.estado, 'Veracruz de Ignacio de la Llave')
        self.assertEqual(escuela.municipio, 'Xalapa')

    def test_mexico_exige_estado_y_municipio(self):
        r = self.registrar(estado_catalogo='', municipio_catalogo='')
        errores = r.context['escuela_form'].errors
        self.assertIn('estado_catalogo', errores)
        self.assertIn('municipio_catalogo', errores)
        self.assertFalse(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_municipio_de_otro_estado_se_rechaza(self):
        otro = Municipio.objects.exclude(estado=self.estado).first()
        r = self.registrar(municipio_catalogo=str(otro.id))
        self.assertIn('municipio_catalogo', r.context['escuela_form'].errors)
        self.assertFalse(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_municipio_inactivo_no_se_acepta(self):
        Municipio.objects.filter(pk=self.municipio.pk).update(activo=False)
        r = self.registrar()
        self.assertIn('municipio_catalogo', r.context['escuela_form'].errors)

    def test_fuera_de_mexico_es_texto_libre(self):
        self.registrar(pais='colombia', clave_centro_trabajo='', estado_catalogo='', municipio_catalogo='',
                       estado='Antioquia', municipio='Medellín')
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        self.assertIsNone(escuela.estado_catalogo)
        self.assertIsNone(escuela.municipio_catalogo)
        self.assertEqual((escuela.estado, escuela.municipio), ('Antioquia', 'Medellín'))

    def test_fuera_de_mexico_exige_texto(self):
        r = self.registrar(pais='colombia', clave_centro_trabajo='', estado_catalogo='', municipio_catalogo='')
        errores = r.context['escuela_form'].errors
        self.assertIn('estado', errores)
        self.assertIn('municipio', errores)

    def test_cambiar_a_otro_pais_limpia_el_catalogo(self):
        self.registrar()
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        datos = datos_escuela(pais='colombia', clave_centro_trabajo='', estado_catalogo='', municipio_catalogo='',
                              estado='Antioquia', municipio='Medellín')
        self.client.post(reverse('escuelas-editar', args=[escuela.id]), datos)
        escuela.refresh_from_db()
        self.assertIsNone(escuela.municipio_catalogo)
        self.assertEqual(escuela.municipio, 'Medellín')

    def test_formulario_ofrece_estados_y_no_todos_los_municipios(self):
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertContains(r, 'Selecciona un estado')
        self.assertContains(r, 'Veracruz de Ignacio de la Llave')
        # Los municipios se piden por estado (AJAX): la página no los trae todos
        self.assertNotContains(r, '>Tijuana<')

    def test_editar_precarga_solo_municipios_de_su_estado(self):
        self.registrar()
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        r = self.client.get(reverse('escuelas-editar', args=[escuela.id]))
        self.assertContains(r, '>Xalapa<')
        self.assertNotContains(r, '>Tijuana<')

    def test_api_municipios(self):
        r = self.client.get(reverse('api-catalogo-municipios', args=[self.estado.id]))
        self.assertEqual(r.status_code, 200)
        nombres = [m['nombre'] for m in r.json()['municipios']]
        self.assertEqual(len(nombres), 212)
        self.assertIn('Xalapa', nombres)

    def test_api_municipios_solo_activos_y_con_login(self):
        Municipio.objects.filter(pk=self.municipio.pk).update(activo=False)
        r = self.client.get(reverse('api-catalogo-municipios', args=[self.estado.id]))
        self.assertNotIn('Xalapa', [m['nombre'] for m in r.json()['municipios']])
        self.client.logout()
        r = self.client.get(reverse('api-catalogo-municipios', args=[self.estado.id]))
        self.assertEqual(r.status_code, 302)

    def test_normalizacion_de_nombres_de_la_migracion(self):
        migracion = importlib.import_module('core.migrations.0009_cargar_catalogo_geo')
        self.assertEqual(migracion.normalizar('Xalapa-Enriquez'), 'xalapa enriquez')
        self.assertEqual(migracion.normalizar('Michoacán de Ocampo'), 'michoacan de ocampo')
        self.assertEqual(migracion.normalizar('  Veracruz '), 'veracruz')


class CargoCatalogoTest(SetUpMixin):

    def setUp(self):
        super().setUp()
        self.client.login(username='admin_t', password='Test1234!')
        self.sin_enlace = Escuela.objects.create(nombre='Sin enlace', municipio='X', estado='Y', nivel='superior')

    def test_los_tres_cargos_se_aceptan(self):
        casos = [('director', 'Director(a)'), ('subdirector', 'Subdirector(a)'), ('coordinador', 'Coordinador(a)')]
        for i, (codigo, etiqueta) in enumerate(casos):
            escuela = Escuela.objects.create(nombre=f'E{i}', municipio='X', estado='Y', nivel='superior')
            self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(
                escuela, nombre=f'Ana{i}', correo=f'ana{i}@itx.edu.mx', cargo=codigo))
            enlace = Usuario.objects.get(email=f'ana{i}@itx.edu.mx')
            self.assertEqual(enlace.get_cargo_display(), etiqueta)

    def test_cargo_fuera_del_catalogo_se_rechaza(self):
        for cargo in ('Coordinadora académica', 'Director General', 'jefe', ''):
            r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace, cargo=cargo))
            self.assertIn('cargo', r.context['form'].errors, cargo)
        self.assertFalse(Usuario.objects.filter(email='mf.hernandez@escuela.edu.mx').exists())

    def test_formulario_ofrece_los_tres_cargos(self):
        r = self.client.get(reverse('escuelas-crear-enlace'))
        for etiqueta in ('Director(a)', 'Subdirector(a)', 'Coordinador(a)'):
            self.assertContains(r, etiqueta)

    def test_editar_precarga_el_cargo(self):
        self.enlace.cargo = 'subdirector'
        self.enlace.save()
        r = self.client.get(reverse('escuelas-editar-enlace', args=[self.enlace.pk]))
        self.assertEqual(r.context['form']['cargo'].value(), 'subdirector')


class SincronizarCatalogoTest(SetUpMixin):

    def datos(self):
        return catalogo_geo.cargar_archivo()

    def test_sin_cambios(self):
        resumen = catalogo_geo.aplicar_catalogo(self.datos())
        for clave, valor in resumen.items():
            if clave != 'detalle':
                self.assertEqual(valor, 0, clave)

    def test_altas_renombres_y_bajas(self):
        datos = self.datos()
        for m in datos['municipios']:
            if m['cvegeo'] == '30087':
                m['nombre'] = 'Xalapa-Enríquez'
        eliminado = datos['municipios'].pop()
        datos['municipios'].append({'cvegeo': '30999', 'cve_ent': '30', 'cve_mun': '999', 'nombre': 'Nuevo Municipio'})
        resumen = catalogo_geo.aplicar_catalogo(datos)
        self.assertEqual((resumen['municipios_nuevos'], resumen['municipios_renombrados'], resumen['municipios_baja']),
                         (1, 1, 1))
        self.assertEqual(Municipio.objects.get(cvegeo='30087').nombre, 'Xalapa-Enríquez')
        self.assertFalse(Municipio.objects.get(cvegeo=eliminado['cvegeo']).activo)
        self.assertTrue(Municipio.objects.get(cvegeo='30999').activo)
        self.assertEqual(Municipio.objects.count(), 2479)  # nada se borra

    def test_baja_se_reactiva_si_vuelve(self):
        datos = self.datos()
        quitado = datos['municipios'].pop()
        catalogo_geo.aplicar_catalogo(datos)
        datos['municipios'].append(quitado)
        resumen = catalogo_geo.aplicar_catalogo(datos)
        self.assertEqual(resumen['municipios_reactivados'], 1)
        self.assertTrue(Municipio.objects.get(cvegeo=quitado['cvegeo']).activo)

    def test_estado_renombrado_actualiza_la_copia_de_las_escuelas(self):
        estado, municipio = veracruz_xalapa()
        escuela = Escuela.objects.create(nombre='E', municipio='Xalapa', estado=estado.nombre, nivel='superior',
                                         estado_catalogo=estado, municipio_catalogo=municipio)
        datos = self.datos()
        for e in datos['estados']:
            if e['cve_ent'] == '30':
                e['nombre'] = 'Veracruz'
        for m in datos['municipios']:
            if m['cvegeo'] == '30087':
                m['nombre'] = 'Xalapa-Enríquez'
        resumen = catalogo_geo.aplicar_catalogo(datos)
        self.assertEqual(resumen['estados_renombrados'], 1)
        escuela.refresh_from_db()
        self.assertEqual((escuela.estado, escuela.municipio), ('Veracruz', 'Xalapa-Enríquez'))

    def test_dry_run_no_guarda(self):
        datos = self.datos()
        datos['municipios'][0]['nombre'] = 'Cambiado'
        resumen = catalogo_geo.aplicar_catalogo(datos, dry_run=True)
        self.assertEqual(resumen['municipios_renombrados'], 1)
        self.assertNotEqual(Municipio.objects.get(cvegeo=datos['municipios'][0]['cvegeo']).nombre, 'Cambiado')

    def test_respuesta_incompleta_no_se_aplica(self):
        datos = self.datos()
        datos['municipios'] = datos['municipios'][:100]
        with self.assertRaises(catalogo_geo.CatalogoInvalido):
            catalogo_geo.aplicar_catalogo(datos)
        datos = self.datos()
        datos['estados'] = datos['estados'][:5]
        with self.assertRaises(catalogo_geo.CatalogoInvalido):
            catalogo_geo.aplicar_catalogo(datos)
        self.assertEqual(Municipio.objects.filter(activo=True).count(), Municipio.objects.count())

    def test_municipio_con_estado_inexistente_se_rechaza(self):
        datos = self.datos()
        datos['municipios'][0]['cve_ent'] = '99'
        with self.assertRaises(catalogo_geo.CatalogoInvalido):
            catalogo_geo.aplicar_catalogo(datos)

    def test_comando_registra_bitacora(self):
        salida = StringIO()
        call_command('sincronizar_catalogo_geo', archivo=str(catalogo_geo.ARCHIVO_RESPALDO), stdout=salida)
        self.assertIn('Catálogo revisado', salida.getvalue())
        self.assertTrue(SincronizacionCatalogo.objects.get().exitosa)

    def test_comando_con_api_caida_falla_y_registra_error(self):
        with mock.patch('core.catalogo_geo.descargar_catalogo', side_effect=Exception('sin red')):
            with self.assertRaises(CommandError):
                call_command('sincronizar_catalogo_geo')
        registro = SincronizacionCatalogo.objects.get()
        self.assertFalse(registro.exitosa)
        self.assertIn('sin red', registro.error)
        self.assertEqual(Municipio.objects.filter(activo=True).count(), Municipio.objects.count())

    def test_comando_dry_run_no_registra_bitacora(self):
        call_command('sincronizar_catalogo_geo', '--dry-run', archivo=str(catalogo_geo.ARCHIVO_RESPALDO),
                     stdout=StringIO())
        self.assertEqual(SincronizacionCatalogo.objects.count(), 0)

    def test_descarga_arma_el_catalogo_desde_la_api(self):
        class Respuesta:
            def __init__(self, datos):
                self._datos = datos

            def raise_for_status(self):
                pass

            def json(self):
                return {'datos': self._datos}

        def get(url, timeout):
            if url.endswith('/mgee/'):
                return Respuesta([{'cve_ent': '01', 'nomgeo': 'Aguascalientes', 'nom_abrev': 'Ags.'}])
            return Respuesta([{'cvegeo': '01001', 'cve_ent': '01', 'cve_mun': '001', 'nomgeo': 'Aguascalientes'}])

        sesion = mock.Mock(headers={}, get=get)
        datos = catalogo_geo.descargar_catalogo(sesion)
        self.assertEqual(datos['estados'], [{'cve_ent': '01', 'nombre': 'Aguascalientes', 'nombre_abreviado': 'Ags.'}])
        self.assertEqual(datos['municipios'][0]['cvegeo'], '01001')

    def test_descarga_reintenta_y_falla_con_error_claro(self):
        sesion = mock.Mock(headers={}, get=mock.Mock(side_effect=catalogo_geo.requests.ConnectionError('caída')))
        with mock.patch('core.catalogo_geo.time.sleep'):
            with self.assertRaises(catalogo_geo.CatalogoInvalido):
                catalogo_geo.descargar_catalogo(sesion)
        self.assertEqual(sesion.get.call_count, catalogo_geo.REINTENTOS)
