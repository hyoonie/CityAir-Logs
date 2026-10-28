from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from core.catalogo_geo import aplicar_catalogo
from core.forms import EscuelaForm
from core.models import Dispositivo, Escuela, Estado, Municipio
from core.tests import SetUpMixin
from core.validators import normalizar_telefono, validar_cct_mexico

PASSWORD = 'Test1234!'


def datos_escuela(**cambios):
    """POST válido de una escuela en Xalapa, Veracruz (catálogo cargado por la migración 0008)."""
    xalapa = Municipio.objects.get(cvegeo='30087')
    datos = {
        'nombre': 'Instituto de Prueba', 'clave_centro_trabajo': '30EPR0001X', 'nivel': 'secundaria',
        'sostenimiento': 'publico', 'turno': 'matutino',
        'telefono': '228 165 0525', 'correo_institucional': 'Contacto@Escuela.edu.mx',
        'pais': 'mexico', 'estado_catalogo': xalapa.estado_id, 'municipio_catalogo': xalapa.id,
        'colonia': 'Centro', 'calle': 'Enríquez', 'numero': '10', 'codigo_postal': '91000',
        'latitud': '19.543800', 'longitud': '-96.910200',
    }
    datos.update(cambios)
    return datos


# ── Validadores ──────────────────────────────────────────────────────────────

class ValidadoresTest(SimpleTestCase):

    def test_cct_valida(self):
        validar_cct_mexico('30EPR0001X')

    def test_cct_invalida(self):
        for clave in ('NE001', '33EPR0001X', '30EP0001X', '30epr0001x', ''):
            with self.subTest(clave=clave), self.assertRaises(ValidationError):
                validar_cct_mexico(clave)

    def test_telefono_se_normaliza(self):
        for valor in ('2281650525', '228 165 0525', '228-165-0525', ' 228-1650525 '):
            with self.subTest(valor=valor):
                self.assertEqual(normalizar_telefono(valor), '228-165-0525')

    def test_telefono_invalido(self):
        for valor in ('12345', '228-165-05250', '(228) 165 0525', 'abc-def-ghij'):
            with self.subTest(valor=valor), self.assertRaises(ValidationError):
                normalizar_telefono(valor)


# ── Formulario ───────────────────────────────────────────────────────────────

class EscuelaFormTest(TestCase):

    def test_formulario_valido_normaliza_y_copia_nombres(self):
        form = EscuelaForm(datos_escuela(clave_centro_trabajo=' 30epr0001x '))
        self.assertTrue(form.is_valid(), form.errors)
        escuela = form.save()
        self.assertEqual(escuela.clave_centro_trabajo, '30EPR0001X')
        self.assertEqual(escuela.telefono, '228-165-0525')
        self.assertEqual(escuela.correo_institucional, 'contacto@escuela.edu.mx')
        self.assertEqual((escuela.estado, escuela.municipio), ('Veracruz de Ignacio de la Llave', 'Xalapa'))

    def test_campos_obligatorios(self):
        form = EscuelaForm({'pais': 'mexico'})
        self.assertFalse(form.is_valid())
        for campo in ('nombre', 'clave_centro_trabajo', 'nivel', 'sostenimiento', 'turno', 'latitud', 'longitud',
                      'estado_catalogo', 'municipio_catalogo'):
            self.assertIn(campo, form.errors)

    def test_turno_obligatorio(self):
        form = EscuelaForm(datos_escuela(turno=''))
        self.assertFalse(form.is_valid())
        self.assertIn('turno', form.errors)

    def test_cct_duplicada_en_el_mismo_pais(self):
        EscuelaForm(datos_escuela()).save()
        form = EscuelaForm(datos_escuela(nombre='Otra'))
        self.assertFalse(form.is_valid())
        self.assertIn('clave_centro_trabajo', form.errors)

    def test_misma_cct_en_otro_pais_es_valida(self):
        EscuelaForm(datos_escuela()).save()
        form = EscuelaForm(datos_escuela(
            pais='colombia', estado='Antioquia', municipio='Medellín',
            estado_catalogo='', municipio_catalogo='',
        ))
        self.assertTrue(form.is_valid(), form.errors)
        escuela = form.save()
        self.assertIsNone(escuela.estado_catalogo)
        self.assertEqual(escuela.municipio, 'Medellín')

    def test_cct_con_formato_libre_fuera_de_mexico(self):
        form = EscuelaForm(datos_escuela(
            pais='colombia', clave_centro_trabajo='105001000001', estado='Antioquia', municipio='Medellín',
        ))
        self.assertTrue(form.is_valid(), form.errors)

    def test_cct_opcional_fuera_de_mexico(self):
        form = EscuelaForm(datos_escuela(
            pais='colombia', clave_centro_trabajo='', estado='Antioquia', municipio='Medellín',
        ))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertIsNone(form.save().clave_centro_trabajo)

    def test_primaria_es_nivel_valido(self):
        form = EscuelaForm(datos_escuela(nivel='primaria'))
        self.assertTrue(form.is_valid(), form.errors)

    def test_nivel_fuera_del_catalogo_no_es_valido(self):
        form = EscuelaForm(datos_escuela(nivel='media_superior'))
        self.assertFalse(form.is_valid())
        self.assertIn('nivel', form.errors)

    def test_cct_invalida_en_mexico(self):
        form = EscuelaForm(datos_escuela(clave_centro_trabajo='NE001'))
        self.assertFalse(form.is_valid())
        self.assertIn('clave_centro_trabajo', form.errors)

    def test_municipio_de_otro_estado(self):
        guadalajara = Municipio.objects.get(cvegeo='14039')
        form = EscuelaForm(datos_escuela(municipio_catalogo=guadalajara.id))
        self.assertFalse(form.is_valid())
        self.assertIn('municipio_catalogo', form.errors)

    def test_otro_pais_exige_estado_y_municipio_de_texto(self):
        form = EscuelaForm(datos_escuela(pais='colombia', estado='', municipio=' '))
        self.assertFalse(form.is_valid())
        self.assertIn('estado', form.errors)
        self.assertIn('municipio', form.errors)

    def test_coordenadas_fuera_de_rango(self):
        form = EscuelaForm(datos_escuela(latitud='91', longitud='-181'))
        self.assertFalse(form.is_valid())
        self.assertIn('latitud', form.errors)
        self.assertIn('longitud', form.errors)

    def test_telefono_y_correo_invalidos(self):
        form = EscuelaForm(datos_escuela(telefono='12345', correo_institucional='no-es-correo'))
        self.assertFalse(form.is_valid())
        self.assertIn('telefono', form.errors)
        self.assertIn('correo_institucional', form.errors)


# ── Endpoints del catálogo ───────────────────────────────────────────────────

class CatalogoApiTest(SetUpMixin):

    def test_requieren_login(self):
        r = self.client.get(reverse('api-catalogo-estados'))
        self.assertEqual(r.status_code, 302)

    def test_estados_activos(self):
        Estado.objects.filter(cve_ent='01').update(activo=False)
        self.client.login(username='general_t', password=PASSWORD)
        estados = self.client.get(reverse('api-catalogo-estados')).json()['estados']
        self.assertEqual(len(estados), 31)
        self.assertNotIn('Aguascalientes', [e['nombre'] for e in estados])

    def test_municipios_por_estado(self):
        veracruz = Estado.objects.get(cve_ent='30')
        Municipio.objects.filter(cvegeo='30118').update(activo=False)
        self.client.login(username='general_t', password=PASSWORD)
        r = self.client.get(reverse('api-catalogo-municipios', args=[veracruz.id]))
        nombres = [m['nombre'] for m in r.json()['municipios']]
        self.assertIn('Xalapa', nombres)
        self.assertNotIn('Orizaba', nombres)
        self.assertEqual(len(nombres), veracruz.municipios.count() - 1)


# ── Registro, edición, listado, detalle, baja y reactivación ────────────────

class GestionEscuelaVistasTest(SetUpMixin):

    def setUp(self):
        super().setUp()
        self.sensor = Dispositivo.objects.create(
            idDispositivo='SENSOR-1', pais='México', estado='Veracruz', ciudad='Xalapa',
            calle='X', colonia='Y', codigo_postal='91000', numero='1', escuela=self.escuela,
        )

    def entrar(self, usuario):
        self.client.login(username=usuario, password=PASSWORD)

    def test_registro_invalido_no_crea_escuela(self):
        self.entrar('admin_t')
        r = self.client.post(reverse('escuelas-registrar'), datos_escuela(clave_centro_trabajo='NE001'))
        self.assertEqual(r.status_code, 200)
        self.assertFalse(Escuela.objects.filter(nombre='Instituto de Prueba').exists())

    def test_formulario_de_registro(self):
        self.entrar('admin_t')
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertContains(r, 'Registrar escuela')
        self.assertContains(r, 'Guardar registro')
        self.assertContains(r, 'Cancelar')
        self.assertContains(r, 'id="mapa-escuela"')
        for texto in ('Selecciona el nivel educativo', 'Selecciona el sostenimiento',
                      'Selecciona el turno', 'Selecciona un país'):
            self.assertContains(r, texto)
        self.assertNotContains(r, '---------')

    def test_formulario_sin_burbujas_y_con_catalogo_de_mensajes(self):
        self.entrar('admin_t')
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertContains(r, 'novalidate')
        self.assertContains(r, '<script id="mensajes-escuela" type="application/json">')
        self.assertContains(r, 'id="error-nombre"')
        self.assertContains(r, 'Ingresa el nombre oficial de la escuela.')

    def test_post_vacio_muestra_mensaje_bajo_cada_obligatorio(self):
        self.entrar('admin_t')
        r = self.client.post(reverse('escuelas-registrar'), {'pais': 'mexico'})
        esperados = {
            'nombre': 'Ingresa el nombre oficial de la escuela.',
            'clave_centro_trabajo': 'Ingresa la clave CCT; es obligatoria para escuelas de México.',
            'nivel': 'Selecciona el nivel educativo.',
            'sostenimiento': 'Selecciona el sostenimiento.',
            'turno': 'Selecciona el turno.',
            'estado_catalogo': 'Selecciona el estado.',
            'municipio_catalogo': 'Selecciona el municipio.',
            'latitud': 'Indica la latitud; puedes elegirla en el mapa.',
            'longitud': 'Indica la longitud; puedes elegirla en el mapa.',
        }
        for campo, mensaje in esperados.items():
            with self.subTest(campo=campo):
                self.assertContains(r, f'<div class="invalid-feedback d-block" id="error-{campo}">{mensaje}</div>',
                                    html=False)

    def test_mensajes_de_formato(self):
        self.entrar('admin_t')
        casos = [
            ({'correo_institucional': 'x@'}, 'correo_institucional', 'Ingresa un correo válido, por ejemplo contacto@escuela.edu.mx.'),
            ({'latitud': 'abc'}, 'latitud', 'Ingresa un número válido.'),
            ({'longitud': '-181'}, 'longitud', 'La longitud debe estar entre -180 y 180.'),
            ({'codigo_postal': '91a00'}, 'codigo_postal', 'El código postal solo debe contener dígitos.'),
            ({'pais': 'colombia', 'estado': ''}, 'estado', 'Ingresa el estado o departamento.'),
        ]
        for cambios, campo, mensaje in casos:
            with self.subTest(campo=campo):
                r = self.client.post(reverse('escuelas-registrar'), datos_escuela(**cambios))
                self.assertContains(r, f'id="error-{campo}">{mensaje}</div>')

    def test_registro_invalido_marca_campos(self):
        self.entrar('admin_t')
        r = self.client.post(reverse('escuelas-registrar'), datos_escuela(telefono='123'))
        self.assertContains(r, 'is-invalid')
        self.assertContains(r, 'El teléfono debe tener 10 dígitos')

    def test_editar_precarga_y_guarda(self):
        self.entrar('admin_t')
        escuela = EscuelaForm(datos_escuela()).save()
        r = self.client.get(reverse('escuelas-editar', args=[escuela.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context['form'].instance, escuela)
        self.assertTrue(r.context['modo_edicion'])
        self.assertContains(r, 'Guardar cambios')
        self.assertContains(r, 'value="Instituto de Prueba"')
        self.assertContains(r, 'value="19.543800"')

        r = self.client.post(reverse('escuelas-editar', args=[escuela.pk]),
                             datos_escuela(nombre='Nombre Editado', telefono='2289998877'))
        self.assertRedirects(r, reverse('escuelas-detalle', args=[escuela.pk]))
        escuela.refresh_from_db()
        self.assertEqual(escuela.nombre, 'Nombre Editado')
        self.assertEqual(escuela.telefono, '228-999-8877')

    def test_editar_conserva_su_propia_cct(self):
        self.entrar('admin_t')
        escuela = EscuelaForm(datos_escuela()).save()
        r = self.client.post(reverse('escuelas-editar', args=[escuela.pk]), datos_escuela(nivel='universidad'))
        self.assertRedirects(r, reverse('escuelas-detalle', args=[escuela.pk]))

    def test_editar_escuela_primaria_sin_cambiar_nivel(self):
        self.entrar('admin_t')
        escuela = EscuelaForm(datos_escuela(nivel='primaria')).save()
        r = self.client.post(reverse('escuelas-editar', args=[escuela.pk]),
                             datos_escuela(nivel='primaria', nombre='Primaria Editada'))
        self.assertRedirects(r, reverse('escuelas-detalle', args=[escuela.pk]))
        escuela.refresh_from_db()
        self.assertEqual(escuela.nivel, 'primaria')
        self.assertEqual(escuela.get_nivel_display(), 'Primaria')

    def test_enlace_no_puede_editar(self):
        self.entrar('enlace_t')
        r = self.client.get(reverse('escuelas-editar', args=[self.escuela.pk]))
        self.assertEqual(r.status_code, 403)

    def test_listado_alfabetico(self):
        Escuela.objects.create(nombre='Alfa', municipio='X', estado='Y', nivel='secundaria')
        Escuela.objects.create(nombre='zeta', municipio='X', estado='Y', nivel='secundaria')
        self.entrar('general_t')
        nombres = [e.nombre for e in self.client.get(reverse('escuelas')).context['escuelas']]
        self.assertEqual(nombres, ['Alfa', 'Escuela Test', 'zeta'])

    def test_baja_y_reactivacion(self):
        self.entrar('admin_t')
        r = self.client.post(reverse('escuelas-baja', args=[self.escuela.pk]))
        self.assertRedirects(r, reverse('escuelas-detalle', args=[self.escuela.pk]))
        self.escuela.refresh_from_db()
        self.assertFalse(self.escuela.activa)

        # Solo la escuela: sensores y usuarios no cambian
        self.sensor.refresh_from_db()
        self.enlace.refresh_from_db()
        self.assertEqual(self.sensor.escuela, self.escuela)
        self.assertEqual(self.enlace.escuela, self.escuela)
        self.assertTrue(self.enlace.is_active)

        # El administrador la sigue viendo; el usuario general no
        self.assertIn(self.escuela, self.client.get(reverse('escuelas')).context['escuelas'])
        self.client.logout()
        self.entrar('general_t')
        self.assertNotIn(self.escuela, self.client.get(reverse('escuelas')).context['escuelas'])
        self.assertEqual(self.client.get(reverse('escuelas-detalle', args=[self.escuela.pk])).status_code, 404)

        self.client.logout()
        self.entrar('admin_t')
        self.client.post(reverse('escuelas-reactivar', args=[self.escuela.pk]))
        self.escuela.refresh_from_db()
        self.assertTrue(self.escuela.activa)

    def test_botones_de_baja_en_detalle(self):
        self.entrar('admin_t')
        url = reverse('escuelas-detalle', args=[self.escuela.pk])
        r = self.client.get(url)
        self.assertContains(r, 'Editar escuela')
        self.assertContains(r, 'Dar de baja')
        self.escuela.activa = False
        self.escuela.save()
        r = self.client.get(url)
        self.assertContains(r, 'Reactivar')
        self.assertContains(r, 'Esta escuela está dada de baja')

    def test_usuario_general_no_ve_botones_de_gestion(self):
        self.entrar('general_t')
        r = self.client.get(reverse('escuelas-detalle', args=[self.escuela.pk]))
        self.assertNotContains(r, 'Editar escuela')
        self.assertNotContains(r, 'Dar de baja')

    def test_baja_solo_por_post(self):
        self.entrar('admin_t')
        r = self.client.get(reverse('escuelas-baja', args=[self.escuela.pk]))
        self.assertEqual(r.status_code, 405)

    def test_enlace_no_puede_dar_de_baja(self):
        self.entrar('enlace_t')
        r = self.client.post(reverse('escuelas-baja', args=[self.escuela.pk]))
        self.assertEqual(r.status_code, 403)
        self.escuela.refresh_from_db()
        self.assertTrue(self.escuela.activa)


# ── Sincronización del catálogo ──────────────────────────────────────────────

class AplicarCatalogoTest(TestCase):

    def setUp(self):
        Municipio.objects.all().delete()
        Estado.objects.all().delete()
        self.minimos = {'minimo_estados': 1, 'minimo_municipios': 1}
        aplicar_catalogo({
            'estados': [{'cve_ent': '30', 'nombre': 'Veracruz'}],
            'municipios': [
                {'cvegeo': '30087', 'cve_ent': '30', 'cve_mun': '087', 'nombre': 'Xalapa-Enríquez'},
                {'cvegeo': '30118', 'cve_ent': '30', 'cve_mun': '118', 'nombre': 'Orizaba'},
            ],
        }, **self.minimos)

    def test_renombra_da_de_baja_y_refresca_copias(self):
        xalapa = Municipio.objects.get(cvegeo='30087')
        escuela = Escuela.objects.create(
            nombre='E', nivel='secundaria', pais='mexico', estado_catalogo=xalapa.estado,
            municipio_catalogo=xalapa, estado='Veracruz', municipio='Xalapa-Enríquez',
        )
        resumen = aplicar_catalogo({
            'estados': [{'cve_ent': '30', 'nombre': 'Veracruz de Ignacio de la Llave'}],
            'municipios': [{'cvegeo': '30087', 'cve_ent': '30', 'cve_mun': '087', 'nombre': 'Xalapa'}],
        }, **self.minimos)

        self.assertEqual(resumen['estados_renombrados'], 1)
        self.assertEqual(resumen['municipios_renombrados'], 1)
        self.assertEqual(resumen['municipios_baja'], 1)
        self.assertFalse(Municipio.objects.get(cvegeo='30118').activo)  # no se borra
        escuela.refresh_from_db()
        self.assertEqual((escuela.estado, escuela.municipio), ('Veracruz de Ignacio de la Llave', 'Xalapa'))

    def test_dry_run_no_guarda(self):
        aplicar_catalogo({
            'estados': [{'cve_ent': '30', 'nombre': 'Otro'}],
            'municipios': [{'cvegeo': '30087', 'cve_ent': '30', 'cve_mun': '087', 'nombre': 'Otro'}],
        }, dry_run=True, **self.minimos)
        self.assertEqual(Estado.objects.get(cve_ent='30').nombre, 'Veracruz')

    def test_rechaza_respuesta_incompleta(self):
        from core.catalogo_geo import CatalogoInvalido
        with self.assertRaises(CatalogoInvalido):
            aplicar_catalogo({'estados': [], 'municipios': []})
