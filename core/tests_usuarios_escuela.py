"""
core/tests_usuarios_escuela.py
Suite de pruebas para la gestión de Docentes y Alumnos por el rol Enlace.

Cubre:
  - Módulo de servicio (validar_persona, validar_filas, crear_usuarios)
  - Permisos en todos los endpoints
  - Flujo completo Docente (A), Alumno individual, Lote (C)
  - Lista con búsqueda/filtros/paginación (D)
  - Acciones: activar/desactivar/reset password
  - Rendimiento: 50 alumnos < 5 s
"""

import json
import time

from django.core.exceptions import ValidationError
from django.test import Client, TestCase
from django.urls import reverse

from core.models import Escuela, TipoUsuario, Usuario
from core.usuarios_escuela import (
    MAX_FILAS_LOTE,
    ROL_ALUMNO,
    ROL_DOCENTE,
    crear_usuarios,
    enlace_de_escuela,
    validar_filas,
    validar_persona,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de fixture
# ──────────────────────────────────────────────────────────────────────────────

def crear_tipo(id_tipo, nombre):
    return TipoUsuario.objects.get_or_create(id_tipo=id_tipo, defaults={'nombre_tipo': nombre})[0]


def crear_usuario(usuario, tipo_id, escuela=None, superuser=False):
    tipo = TipoUsuario.objects.get(id_tipo=tipo_id)
    u = Usuario.objects.create_user(
        usuario=usuario, email=f'{usuario}@test.mx',
        password='Test1234!', nombre='Test', aPaterno='User', aMaterno=''
    )
    u.tipousuario = tipo
    u.escuela = escuela
    u.is_superuser = superuser
    u.is_staff = superuser
    u.save()
    return u


def fila(i, escuela_id=1, dominio='escuela.edu.mx'):
    return {
        'nombre': f'Alumno{i}',
        'apellidos': f'Test{i} Prueba',
        'correo': f'alumno{i}_{escuela_id}@{dominio}',
        'telefono': '',
    }


class BaseTest(TestCase):
    def setUp(self):
        crear_tipo(1, 'Administrador de sistema')
        crear_tipo(3, 'Usuario general')
        crear_tipo(4, 'Enlace')
        crear_tipo(5, 'Docente')
        crear_tipo(6, 'Alumno')

        self.escuela = Escuela.objects.create(
            nombre='Escuela Test', municipio='Xalapa',
            estado='Veracruz', nivel='superior', activa=True,
        )
        self.otra_escuela = Escuela.objects.create(
            nombre='Otra Escuela', municipio='Veracruz',
            estado='Veracruz', nivel='secundaria', activa=True,
        )

        self.enlace  = crear_usuario('enlace_t',  4, escuela=self.escuela)
        self.enlace2 = crear_usuario('enlace2_t', 4, escuela=self.otra_escuela)
        self.docente = crear_usuario('docente_t', 5, escuela=self.escuela)
        self.alumno  = crear_usuario('alumno_t',  6, escuela=self.escuela)
        self.admin   = crear_usuario('admin_t',   1, superuser=True)
        self.client  = Client()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Servicio: validar_persona
# ──────────────────────────────────────────────────────────────────────────────

class ValidarPersonaTest(BaseTest):

    def test_fila_valida(self):
        limpios = validar_persona({
            'nombre': 'María', 'apellidos': 'López García',
            'correo': 'maria@escuela.edu.mx', 'telefono': '228 165 0525',
        })
        self.assertEqual(limpios['nombre'], 'María')
        self.assertEqual(limpios['aPaterno'], 'López')
        self.assertEqual(limpios['aMaterno'], 'García')
        self.assertEqual(limpios['correo'], 'maria@escuela.edu.mx')
        self.assertEqual(limpios['telefono'], '228-165-0525')

    def test_correo_se_normaliza_a_minusculas(self):
        limpios = validar_persona({
            'nombre': 'Ana', 'apellidos': 'Cruz',
            'correo': 'ANA@ESCUELA.EDU.MX', 'telefono': '',
        })
        self.assertEqual(limpios['correo'], 'ana@escuela.edu.mx')

    def test_telefono_opcional(self):
        limpios = validar_persona({
            'nombre': 'Ana', 'apellidos': 'Cruz',
            'correo': 'ana@gmail.com', 'telefono': '',
        })
        self.assertIsNone(limpios['telefono'])

    def test_nombre_obligatorio(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': '', 'apellidos': 'X', 'correo': 'x@gmail.com', 'telefono': ''})
        self.assertIn('nombre', ctx.exception.message_dict)

    def test_apellidos_obligatorio(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': 'X', 'apellidos': '', 'correo': 'x@gmail.com', 'telefono': ''})
        self.assertIn('apellidos', ctx.exception.message_dict)

    def test_correo_formato_invalido(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': 'X', 'apellidos': 'Y', 'correo': 'no-es-correo', 'telefono': ''})
        self.assertIn('correo', ctx.exception.message_dict)

    def test_correo_dominio_no_permitido(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': 'X', 'apellidos': 'Y', 'correo': 'x@hotmail.com', 'telefono': ''})
        self.assertIn('correo', ctx.exception.message_dict)

    def test_telefono_formato_invalido(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': 'X', 'apellidos': 'Y', 'correo': 'x@gmail.com', 'telefono': '12345'})
        self.assertIn('telefono', ctx.exception.message_dict)

    def test_multiples_errores_devueltos(self):
        with self.assertRaises(ValidationError) as ctx:
            validar_persona({'nombre': '', 'apellidos': '', 'correo': 'bad', 'telefono': 'bad'})
        errs = ctx.exception.message_dict
        self.assertIn('nombre', errs)
        self.assertIn('apellidos', errs)
        self.assertIn('correo', errs)
        self.assertIn('telefono', errs)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Servicio: validar_filas
# ──────────────────────────────────────────────────────────────────────────────

class ValidarFilasTest(BaseTest):

    def test_filas_validas_sin_errores(self):
        filas = [fila(i, self.escuela.id) for i in range(5)]
        errores, limpias = validar_filas(filas)
        self.assertEqual(errores, [])
        self.assertEqual(len(limpias), 5)
        self.assertTrue(all(f is not None for f in limpias))

    def test_correo_duplicado_dentro_del_lote(self):
        filas = [
            {'nombre': 'A', 'apellidos': 'X', 'correo': 'dup@gmail.com', 'telefono': ''},
            {'nombre': 'B', 'apellidos': 'Y', 'correo': 'dup@gmail.com', 'telefono': ''},
        ]
        errores, _ = validar_filas(filas)
        self.assertTrue(any(e['campo'] == 'correo' and e['fila'] == 2 for e in errores))

    def test_correo_duplicado_en_bd(self):
        self.alumno.email = 'existente@gmail.com'
        self.alumno.save()
        filas = [{'nombre': 'X', 'apellidos': 'Y', 'correo': 'existente@gmail.com', 'telefono': ''}]
        errores, _ = validar_filas(filas)
        self.assertTrue(any(e['campo'] == 'correo' for e in errores))

    def test_correo_duplicado_bd_case_insensitive(self):
        self.alumno.email = 'existente@gmail.com'
        self.alumno.save()
        filas = [{'nombre': 'X', 'apellidos': 'Y', 'correo': 'EXISTENTE@gmail.com', 'telefono': ''}]
        errores, _ = validar_filas(filas)
        self.assertTrue(any(e['campo'] == 'correo' for e in errores))

    def test_fila_invalida_no_afecta_las_validas(self):
        filas = [
            {'nombre': 'OK', 'apellidos': 'Apellido', 'correo': 'ok@gmail.com', 'telefono': ''},
            {'nombre': '',   'apellidos': 'X',        'correo': 'bad',           'telefono': ''},
        ]
        errores, limpias = validar_filas(filas)
        self.assertTrue(len(errores) > 0)
        # La fila 1 está limpia, la 2 marcada como None
        self.assertIsNotNone(limpias[0])
        self.assertIsNone(limpias[1])

    def test_filas_limpias_confiables_cuando_sin_errores(self):
        filas = [fila(i, self.escuela.id) for i in range(3)]
        errores, limpias = validar_filas(filas)
        self.assertEqual(errores, [])
        for l in limpias:
            self.assertIn('correo', l)
            self.assertIn('aPaterno', l)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Servicio: crear_usuarios
# ──────────────────────────────────────────────────────────────────────────────

class CrearUsuariosTest(BaseTest):

    def _filas_limpias(self, n, rol=ROL_ALUMNO):
        raw = [fila(i + 1000, self.escuela.id) for i in range(n)]
        _, limpias = validar_filas(raw)
        return limpias

    def test_crea_usuarios_con_rol_y_escuela_correctos(self):
        limpias = self._filas_limpias(3)
        resultado = crear_usuarios(self.escuela, ROL_ALUMNO, limpias)
        self.assertEqual(len(resultado), 3)
        for r in resultado:
            u = Usuario.objects.get(email=r['correo'])
            self.assertEqual(u.tipousuario_id, ROL_ALUMNO)
            self.assertEqual(u.escuela_id, self.escuela.id)
            self.assertTrue(u.debe_cambiar_password)
            self.assertTrue(u.is_active)

    def test_sin_usuarios_huerfanos(self):
        limpias = self._filas_limpias(5)
        crear_usuarios(self.escuela, ROL_ALUMNO, limpias)
        huerfanos = Usuario.objects.filter(tipousuario_id=ROL_ALUMNO, escuela__isnull=True)
        self.assertEqual(huerfanos.count(), 0)

    def test_logins_unicos_con_mismo_nombre(self):
        """50 alumnos con el mismo primer nombre generan 50 usuarios distintos."""
        filas_raw = [
            {'nombre': 'María', 'apellidos': f'Pérez{i}', 'correo': f'maria{i}@gmail.com', 'telefono': ''}
            for i in range(50)
        ]
        _, limpias = validar_filas(filas_raw)
        resultado = crear_usuarios(self.escuela, ROL_ALUMNO, limpias)
        usuarios = [r['usuario'] for r in resultado]
        self.assertEqual(len(usuarios), len(set(usuarios)))

    def test_hash_ligero_permite_login(self):
        """El usuario creado con hash ligero puede iniciar sesión."""
        limpias = self._filas_limpias(1)
        resultado = crear_usuarios(self.escuela, ROL_ALUMNO, limpias)
        r = resultado[0]
        ok = self.client.login(username=r['correo'], password=r['password'])
        self.assertTrue(ok)

    def test_rendimiento_50_alumnos_en_menos_de_5s(self):
        """Criterio de rendimiento: 50 alumnos creados en menos de 5 segundos."""
        filas_raw = [fila(i + 5000, self.escuela.id) for i in range(50)]
        _, limpias = validar_filas(filas_raw)
        t0 = time.monotonic()
        resultado = crear_usuarios(self.escuela, ROL_ALUMNO, limpias)
        elapsed = time.monotonic() - t0
        self.assertEqual(len(resultado), 50)
        self.assertLess(elapsed, 5.0, f'Tardó {elapsed:.2f}s, límite es 5s.')

    def test_docente_con_rol_correcto(self):
        raw = [{'nombre': 'Prof', 'apellidos': 'García', 'correo': 'prof@gmail.com', 'telefono': ''}]
        _, limpias = validar_filas(raw)
        crear_usuarios(self.escuela, ROL_DOCENTE, limpias)
        u = Usuario.objects.get(email='prof@gmail.com')
        self.assertEqual(u.tipousuario_id, ROL_DOCENTE)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Permiso: enlace_de_escuela
# ──────────────────────────────────────────────────────────────────────────────

class EnlaceDeEscuelaTest(BaseTest):

    def test_enlace_con_escuela_activa(self):
        self.assertIsNotNone(enlace_de_escuela(self.enlace))

    def test_docente_devuelve_none(self):
        self.assertIsNone(enlace_de_escuela(self.docente))

    def test_alumno_devuelve_none(self):
        self.assertIsNone(enlace_de_escuela(self.alumno))

    def test_superuser_devuelve_none(self):
        self.assertIsNone(enlace_de_escuela(self.admin))

    def test_enlace_sin_escuela_devuelve_none(self):
        sin_escuela = crear_usuario('enlace_noesc', 4)
        self.assertIsNone(enlace_de_escuela(sin_escuela))

    def test_enlace_con_escuela_inactiva_devuelve_none(self):
        self.escuela.activa = False
        self.escuela.save()
        self.enlace.refresh_from_db()
        result = enlace_de_escuela(self.enlace)
        self.assertIsNone(result)
        self.escuela.activa = True
        self.escuela.save()


# ──────────────────────────────────────────────────────────────────────────────
# 5. API: Crear docente
# ──────────────────────────────────────────────────────────────────────────────

def payload_persona(i=1, dominio='gmail.com'):
    return {
        'nombre': f'Persona{i}',
        'apellidos': f'Test{i} Prueba',
        'correo': f'persona{i}@{dominio}',
        'telefono': '',
    }


class ApiDocenteTest(BaseTest):

    URL = 'api-escuelas-docente-crear'

    def _post(self, datos, usuario=None):
        if usuario:
            self.client.force_login(usuario)
        return self.client.post(
            reverse(self.URL),
            data=json.dumps(datos),
            content_type='application/json',
        )

    def test_enlace_crea_docente_exitosamente(self):
        r = self._post(payload_persona(1), self.enlace)
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.content)
        self.assertIn('creado', data)
        u = Usuario.objects.get(email='persona1@gmail.com')
        self.assertEqual(u.tipousuario_id, ROL_DOCENTE)
        self.assertEqual(u.escuela_id, self.escuela.id)
        self.assertTrue(u.debe_cambiar_password)

    def test_docente_recibe_403(self):
        r = self._post(payload_persona(2), self.docente)
        self.assertEqual(r.status_code, 403)

    def test_alumno_recibe_403(self):
        r = self._post(payload_persona(3), self.alumno)
        self.assertEqual(r.status_code, 403)

    def test_anonimo_redirige_a_login(self):
        r = self.client.post(reverse(self.URL), data='{}', content_type='application/json')
        self.assertEqual(r.status_code, 302)
        self.assertIn('/login', r['Location'])

    def test_superuser_recibe_403(self):
        r = self._post(payload_persona(4), self.admin)
        self.assertEqual(r.status_code, 403)

    def test_correo_duplicado_devuelve_400(self):
        self._post(payload_persona(5), self.enlace)  # primer registro
        r = self._post(payload_persona(5), self.enlace)  # duplicado
        self.assertEqual(r.status_code, 400)

    def test_escuela_id_ajeno_devuelve_403(self):
        datos = payload_persona(6)
        datos['escuela_id'] = self.otra_escuela.id
        r = self._post(datos, self.enlace)
        self.assertEqual(r.status_code, 403)
        self.assertFalse(Usuario.objects.filter(email='persona6@gmail.com').exists())

    def test_escuela_id_propio_funciona(self):
        datos = payload_persona(7)
        datos['escuela_id'] = self.escuela.id
        r = self._post(datos, self.enlace)
        self.assertEqual(r.status_code, 201)

    def test_dominio_no_permitido_devuelve_400(self):
        datos = payload_persona(8, dominio='hotmail.com')
        r = self._post(datos, self.enlace)
        self.assertEqual(r.status_code, 400)


# ──────────────────────────────────────────────────────────────────────────────
# 6. API: Crear alumno individual
# ──────────────────────────────────────────────────────────────────────────────

class ApiAlumnoTest(BaseTest):

    URL = 'api-escuelas-alumno-crear'

    def test_enlace_crea_alumno_exitosamente(self):
        self.client.force_login(self.enlace)
        r = self.client.post(
            reverse(self.URL),
            data=json.dumps(payload_persona(10)),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 201)
        u = Usuario.objects.get(email='persona10@gmail.com')
        self.assertEqual(u.tipousuario_id, ROL_ALUMNO)
        self.assertEqual(u.escuela_id, self.escuela.id)

    def test_docente_recibe_403(self):
        self.client.force_login(self.docente)
        r = self.client.post(
            reverse(self.URL),
            data=json.dumps(payload_persona(11)),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 403)


# ──────────────────────────────────────────────────────────────────────────────
# 7. API: Lote (todo o nada)
# ──────────────────────────────────────────────────────────────────────────────

class ApiAlumnosLoteTest(BaseTest):

    URL = 'api-escuelas-alumnos-lote'

    def _post(self, filas, usuario=None):
        if usuario:
            self.client.force_login(usuario)
        return self.client.post(
            reverse(self.URL),
            data=json.dumps({'filas': filas}),
            content_type='application/json',
        )

    def test_lote_de_50_se_crea_correctamente(self):
        filas_raw = [fila(i, self.escuela.id) for i in range(50)]
        r = self._post(filas_raw, self.enlace)
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.content)
        self.assertEqual(data['creados'], 50)
        self.assertEqual(len(data['usuarios']), 50)
        # Verificar en BD
        qs = Usuario.objects.filter(escuela=self.escuela, tipousuario_id=ROL_ALUMNO)
        # -1 por el alumno del setUp
        self.assertGreaterEqual(qs.count(), 50)

    def test_lote_vacio_devuelve_400(self):
        r = self._post([], self.enlace)
        self.assertEqual(r.status_code, 400)

    def test_lote_mas_de_200_devuelve_400(self):
        filas_raw = [fila(i, self.escuela.id) for i in range(201)]
        r = self._post(filas_raw, self.enlace)
        self.assertEqual(r.status_code, 400)

    def test_todo_o_nada_fila_invalida_no_crea_nada(self):
        filas_raw = [fila(i, self.escuela.id) for i in range(5)]
        filas_raw.append({'nombre': '', 'apellidos': '', 'correo': 'bad', 'telefono': ''})
        antes = Usuario.objects.filter(escuela=self.escuela, tipousuario_id=ROL_ALUMNO).count()
        r = self._post(filas_raw, self.enlace)
        self.assertEqual(r.status_code, 400)
        despues = Usuario.objects.filter(escuela=self.escuela, tipousuario_id=ROL_ALUMNO).count()
        self.assertEqual(antes, despues)

    def test_escuela_id_ajeno_devuelve_403(self):
        self.client.force_login(self.enlace)
        r = self.client.post(
            reverse(self.URL),
            data=json.dumps({'filas': [fila(99, self.escuela.id)], 'escuela_id': self.otra_escuela.id}),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 403)

    def test_docente_recibe_403(self):
        r = self._post([fila(99, self.escuela.id)], self.docente)
        self.assertEqual(r.status_code, 403)

    def test_sin_duplicados_ni_huerfanos(self):
        filas_raw = [fila(i + 200, self.escuela.id) for i in range(10)]
        self._post(filas_raw, self.enlace)
        # Ninguno sin escuela
        self.assertEqual(
            Usuario.objects.filter(tipousuario_id=ROL_ALUMNO, escuela__isnull=True).count(), 0
        )
        # Todos con la escuela correcta
        correos = {f['correo'] for f in filas_raw}
        for correo in correos:
            u = Usuario.objects.get(email=correo)
            self.assertEqual(u.escuela_id, self.escuela.id)


# ──────────────────────────────────────────────────────────────────────────────
# 8. API: Validar sin guardar
# ──────────────────────────────────────────────────────────────────────────────

class ApiAlumnosValidarTest(BaseTest):

    URL = 'api-escuelas-alumnos-validar'

    def test_filas_validas_sin_errores(self):
        self.client.force_login(self.enlace)
        filas_raw = [fila(i + 300, self.escuela.id) for i in range(3)]
        r = self.client.post(
            reverse(self.URL),
            data=json.dumps({'filas': filas_raw}),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertEqual(data['errores'], [])

    def test_correo_existente_aparece_en_errores(self):
        self.client.force_login(self.enlace)
        filas_raw = [{'nombre': 'X', 'apellidos': 'Y',
                      'correo': self.alumno.email, 'telefono': ''}]
        r = self.client.post(
            reverse(self.URL),
            data=json.dumps({'filas': filas_raw}),
            content_type='application/json',
        )
        data = json.loads(r.content)
        self.assertTrue(len(data['errores']) > 0)

    def test_no_crea_usuarios(self):
        self.client.force_login(self.enlace)
        filas_raw = [fila(400, self.escuela.id)]
        antes = Usuario.objects.count()
        self.client.post(
            reverse(self.URL),
            data=json.dumps({'filas': filas_raw}),
            content_type='application/json',
        )
        self.assertEqual(Usuario.objects.count(), antes)


# ──────────────────────────────────────────────────────────────────────────────
# 9. API: Reset de contraseña temporal
# ──────────────────────────────────────────────────────────────────────────────

class ApiResetPasswordTest(BaseTest):

    def _url(self, user_id):
        return reverse('api-escuelas-reset-password', args=[user_id])

    def test_reset_docente_propio_exitoso(self):
        self.client.force_login(self.enlace)
        r = self.client.post(self._url(self.docente.pk), content_type='application/json')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn('password', data)
        # Login con la nueva contraseña
        self.client.logout()
        ok = self.client.login(username=self.docente.email, password=data['password'])
        self.assertTrue(ok)

    def test_reset_usuario_otra_escuela_devuelve_404(self):
        # docente de otra escuela
        docente2 = crear_usuario('doc_otra', 5, escuela=self.otra_escuela)
        self.client.force_login(self.enlace)
        r = self.client.post(self._url(docente2.pk), content_type='application/json')
        self.assertEqual(r.status_code, 404)

    def test_docente_no_puede_resetear_a_nadie(self):
        self.client.force_login(self.docente)
        r = self.client.post(self._url(self.alumno.pk), content_type='application/json')
        self.assertEqual(r.status_code, 403)

    def test_enlace_no_puede_resetear_a_otro_enlace(self):
        # El Enlace solo puede resetear Docente/Alumno, no a otro Enlace
        self.client.force_login(self.enlace)
        # Crear otro Enlace en la misma escuela (no debería poder hacer reset)
        enlace_extra = crear_usuario('enlace_extra', 4, escuela=self.escuela)
        r = self.client.post(self._url(enlace_extra.pk), content_type='application/json')
        self.assertEqual(r.status_code, 404)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Lista (Actividad D): búsqueda, filtros, paginación
# ──────────────────────────────────────────────────────────────────────────────

class ListaUsuariosTest(BaseTest):

    def setUp(self):
        super().setUp()
        # Crear 120 usuarios (60 Docentes, 60 Alumnos) en la escuela del Enlace
        for i in range(60):
            crear_usuario(f'doc_{i}', ROL_DOCENTE, escuela=self.escuela)
        for i in range(60):
            crear_usuario(f'alu_{i}', ROL_ALUMNO, escuela=self.escuela)
        # 30 usuarios en la otra escuela (no deben aparecer)
        for i in range(30):
            crear_usuario(f'otro_{i}', ROL_ALUMNO, escuela=self.otra_escuela)

        self.client.force_login(self.enlace)

    def test_solo_muestra_usuarios_de_su_escuela(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        self.assertEqual(r.status_code, 200)
        # Ningún usuario de otra escuela debe aparecer
        for u in r.context['page_obj'].object_list:
            self.assertEqual(u.escuela_id, self.escuela.id)

    def test_paginacion_25_por_pagina(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        self.assertEqual(len(r.context['page_obj'].object_list), 25)

    def test_total_correcto(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        # 60+60+1(setUp alumno)+1(setUp docente) = 122
        self.assertGreaterEqual(r.context['total'], 120)

    def test_pagina_fuera_de_rango_muestra_ultima(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?page=9999')
        self.assertEqual(r.status_code, 200)
        # Devuelve la última página (no rompe)
        self.assertFalse(r.context['page_obj'].has_next())

    def test_filtro_rol_docente(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?rol=docente')
        for u in r.context['page_obj'].object_list:
            self.assertEqual(u.tipousuario_id, ROL_DOCENTE)

    def test_filtro_rol_alumno(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?rol=alumno')
        for u in r.context['page_obj'].object_list:
            self.assertEqual(u.tipousuario_id, ROL_ALUMNO)

    def test_busqueda_por_nombre(self):
        # Crear usuario con nombre único para la búsqueda
        u_especial = crear_usuario('buscame_t', ROL_DOCENTE, escuela=self.escuela)
        u_especial.nombre = 'Buzcable'
        u_especial.save()
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?q=Buzcable')
        usuarios = list(r.context['page_obj'].object_list)
        self.assertTrue(any(u.pk == u_especial.pk for u in usuarios))

    def test_busqueda_dos_palabras(self):
        u = crear_usuario('busca2_t', ROL_ALUMNO, escuela=self.escuela)
        u.nombre = 'Rosalinda'
        u.aPaterno = 'Zaragoza'
        u.save()
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?q=Rosalinda+Zaragoza')
        usuarios = list(r.context['page_obj'].object_list)
        self.assertTrue(any(u2.pk == u.pk for u2 in usuarios))

    def test_filtros_persisten_en_paginacion(self):
        r = self.client.get(reverse('escuelas-gestionar-usuarios') + '?rol=alumno&estado=activos&page=2')
        self.assertEqual(r.status_code, 200)
        # No lanza error y devuelve página 2
        self.assertIn('page_obj', r.context)

    def test_docente_recibe_403(self):
        self.client.force_login(self.docente)
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        self.assertEqual(r.status_code, 403)

    def test_superuser_recibe_403(self):
        self.client.force_login(self.admin)
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        self.assertEqual(r.status_code, 403)


# ──────────────────────────────────────────────────────────────────────────────
# 11. Acciones: activar/desactivar
# ──────────────────────────────────────────────────────────────────────────────

class AccionesListaTest(BaseTest):

    def setUp(self):
        super().setUp()
        self.client.force_login(self.enlace)

    def test_desactivar_alumno_propio(self):
        r = self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'desactivar', 'user_id': self.alumno.pk,
        })
        self.assertRedirects(r, reverse('escuelas-gestionar-usuarios'), fetch_redirect_response=False)
        self.alumno.refresh_from_db()
        self.assertFalse(self.alumno.is_active)

    def test_activar_alumno_propio(self):
        self.alumno.is_active = False
        self.alumno.save()
        r = self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'activar', 'user_id': self.alumno.pk,
        })
        self.assertRedirects(r, reverse('escuelas-gestionar-usuarios'), fetch_redirect_response=False)
        self.alumno.refresh_from_db()
        self.assertTrue(self.alumno.is_active)

    def test_enlace_no_puede_desactivarse_a_si_mismo(self):
        """El Enlace envía su propio user_id — debe fallar porque no es Docente/Alumno."""
        r = self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'desactivar', 'user_id': self.enlace.pk,
        })
        self.assertRedirects(r, reverse('escuelas-gestionar-usuarios'), fetch_redirect_response=False)
        self.enlace.refresh_from_db()
        self.assertTrue(self.enlace.is_active)  # no se desactivó

    def test_enlace_no_puede_desactivar_usuario_de_otra_escuela(self):
        alumno_otro = crear_usuario('alu_otro', ROL_ALUMNO, escuela=self.otra_escuela)
        self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'desactivar', 'user_id': alumno_otro.pk,
        })
        alumno_otro.refresh_from_db()
        self.assertTrue(alumno_otro.is_active)  # no se desactivó

