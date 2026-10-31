from django.test import TestCase, Client
from django.urls import reverse
from core.models import Usuario, TipoUsuario, Escuela, Dispositivo, Municipio
from django.contrib.messages import get_messages
from core.utils import generar_password_temporal


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


class SetUpMixin(TestCase):
    def setUp(self):
        crear_tipo(1, 'Administrador de sistema')
        crear_tipo(3, 'Usuario general')
        crear_tipo(4, 'Enlace')
        crear_tipo(5, 'Docente')
        crear_tipo(6, 'Alumno')

        self.escuela = Escuela.objects.create(
            nombre='Escuela Test', municipio='Xalapa',
            estado='Veracruz', nivel='universidad', activa=True
        )
        self.admin    = crear_usuario('admin_t',   1, superuser=True)
        self.enlace   = crear_usuario('enlace_t',  4, escuela=self.escuela)
        self.docente  = crear_usuario('docente_t', 5, escuela=self.escuela)
        self.alumno   = crear_usuario('alumno_t',  6, escuela=self.escuela)
        self.general  = crear_usuario('general_t', 3)
        self.client = Client()


# ── Acceso a /escuelas/ ──────────────────────────────────────────────────────

class EscuelasAccesoTest(SetUpMixin):

    def test_redirige_si_no_autenticado(self):
        r = self.client.get(reverse('escuelas'))
        self.assertEqual(r.status_code, 302)
        self.assertIn('/login', r['Location'])

    def test_usuario_general_ve_escuelas(self):
        self.client.login(username='general_t', password='Test1234!')
        r = self.client.get(reverse('escuelas'))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'Programa Escuelas')

    def test_enlace_ve_su_escuela(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas'))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'Escuela Test')

    def test_superusuario_ve_tabla_de_enlaces(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.get(reverse('escuelas'))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'enlace_t')


# ── Crear Enlace (solo superusuario) ─────────────────────────────────────────

class CrearEnlaceTest(SetUpMixin):

    def test_superusuario_puede_crear_enlace(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-crear-enlace'), {
            'email': 'nuevo@test.mx',
            'nombre': 'Nuevo', 'aPaterno': 'Enlace',
            'escuela_id': self.escuela.pk
        })
        # Ya no redirige: muestra las credenciales una sola vez
        self.assertEqual(r.status_code, 200)
        self.assertTemplateUsed(r, 'escuelas/credenciales-enlace.html')

        usuario = Usuario.objects.get(usuario=f'nuevo_{self.escuela.pk}')
        self.assertEqual(usuario.tipousuario_id, Usuario.TIPO_ENLACE)
        self.assertEqual(usuario.escuela, self.escuela)
        self.assertTrue(usuario.debe_cambiar_password)

        # La contraseña mostrada es la que quedó guardada (como hash)
        password = r.context['password_temporal']
        self.assertTrue(usuario.check_password(password))
        self.assertContains(r, password)

        # La contraseña no debe viajar por messages (queda en sesión)
        for m in get_messages(r.wsgi_request):
            self.assertNotIn(password, str(m))

    def test_enlace_no_puede_crear_enlace(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-crear-enlace'))
        self.assertEqual(r.status_code, 403)

    def test_alumno_no_puede_crear_enlace(self):
        self.client.login(username='alumno_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-crear-enlace'))
        self.assertEqual(r.status_code, 403)

class PasswordTemporalTest(SetUpMixin):

    def test_generador_longitud_y_variedad(self):
        p = generar_password_temporal()
        self.assertEqual(len(p), 12)
        self.assertTrue(any(c.isupper() for c in p))
        self.assertTrue(any(c.islower() for c in p))
        self.assertTrue(any(c.isdigit() for c in p))
        self.assertTrue(any(c in '!@#$%*?' for c in p))

    def test_generador_no_repite(self):
        self.assertNotEqual(generar_password_temporal(), generar_password_temporal())

    def test_restablecer_cambia_password_y_marca_bandera(self):
        enlace = Usuario.objects.get(usuario='enlace_t')
        enlace.debe_cambiar_password = False
        enlace.save()
        nueva = enlace.restablecer_password_temporal()
        enlace.refresh_from_db()
        self.assertTrue(enlace.check_password(nueva))
        self.assertFalse(enlace.check_password('Test1234!'))
        self.assertTrue(enlace.debe_cambiar_password)


# ── Generación automática de usuario Enlace ──────────────────────────────────

class GenerarUsuarioEnlaceTest(SetUpMixin):

    def _crear(self, usuario, n):
        # Crea un usuario cualquiera para simular que ese nombre ya está ocupado
        return Usuario.objects.create_user(
            usuario=usuario, email=f'u{n}@test.mx', password='Pass1234!',
            nombre='X', aPaterno='Y', aMaterno='Z',
        )

    def test_formato_basico(self):
        r = Usuario.objects.generar_usuario_enlace('Juan', 2)
        self.assertEqual(r, 'juan_2')

    def test_duplicados_agregan_sufijo(self):
        self._crear('juan_2', 1)
        self.assertEqual(Usuario.objects.generar_usuario_enlace('Juan', 2), 'juan_2-2')
        self._crear('juan_2-2', 2)
        self.assertEqual(Usuario.objects.generar_usuario_enlace('Juan', 2), 'juan_2-3')

    def test_mismo_nombre_en_otra_escuela_no_es_duplicado(self):
        self._crear('juan_2', 1)
        self.assertEqual(Usuario.objects.generar_usuario_enlace('Juan', 3), 'juan_3')

    def test_acentos_y_enie(self):
        self.assertEqual(Usuario.objects.generar_usuario_enlace('José', 2), 'jose_2')
        self.assertEqual(Usuario.objects.generar_usuario_enlace('Muñoz', 2), 'munoz_2')

    def test_nombre_compuesto_usa_solo_el_primero(self):
        self.assertEqual(Usuario.objects.generar_usuario_enlace('María José', 2), 'maria_2')

    def test_nombre_vacio_o_sin_caracteres_validos(self):
        self.assertEqual(Usuario.objects.generar_usuario_enlace('', 2), 'enlace_2')
        self.assertEqual(Usuario.objects.generar_usuario_enlace('李伟', 2), 'enlace_2')

    def test_nombre_largo_respeta_45_caracteres(self):
        r = Usuario.objects.generar_usuario_enlace('a' * 100, 2)
        self.assertLessEqual(len(r), 45)
        self.assertTrue(r.endswith('_2'))

class ApiRegistroEnlaceTest(SetUpMixin):

    def _post(self, **extra):
        datos = {'escuela_id': self.escuela.pk, 'nombre': 'Maria', 'aPaterno': 'Lopez',
                 'email': 'maria@test.mx', 'cargo': 'director'}
        datos.update(extra)
        return self.client.post(reverse('api-registrar-enlace'), datos)

    def test_registro_exitoso(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self._post(telefono='2281650525')
        self.assertEqual(r.status_code, 201)
        u = Usuario.objects.get(usuario=r.json()['usuario'])
        self.assertTrue(u.debe_cambiar_password)
        self.assertEqual(u.telefono, '228-165-0525')
        self.assertTrue(u.check_password(r.json()['password_temporal']))

    def test_segundo_enlace_en_la_misma_escuela(self):
        self.client.login(username='admin_t', password='Test1234!')
        self._post()
        r = self._post(email='otro@test.mx')
        self.assertEqual(r.status_code, 201)
        self.assertTrue(r.json()['usuario'].endswith('-2'))

    def test_errores_de_validacion(self):
        self.client.login(username='admin_t', password='Test1234!')
        self.assertEqual(self._post(cargo='rey').status_code, 400)
        self.assertEqual(self._post(telefono='123').status_code, 400)
        self.assertEqual(self._post(email='no-es-correo').status_code, 400)
        self.assertEqual(self._post(nombre='').status_code, 400)

    def test_correo_duplicado_409(self):
        self.client.login(username='admin_t', password='Test1234!')
        self._post()
        self.assertEqual(self._post().status_code, 409)

    def test_permisos_y_metodo(self):
        url = reverse('api-registrar-enlace')
        self.client.login(username='enlace_t', password='Test1234!')
        self.assertEqual(self.client.post(url).status_code, 403)
        self.client.login(username='admin_t', password='Test1234!')
        self.assertEqual(self.client.get(url).status_code, 405)

    def test_listado_filtra_sin_enlace(self):
        self.client.login(username='admin_t', password='Test1234!')
        url = reverse('api-escuelas-para-enlace')
        todas = self.client.get(url).json()['escuelas']
        sin = self.client.get(url, {'sin_enlace': '1'}).json()['escuelas']
        self.assertTrue(all(e['enlaces_activos'] == 0 for e in sin))
        self.assertLessEqual(len(sin), len(todas))


# ── Gestionar usuarios (Enlace) ───────────────────────────────────────────────

class GestionarUsuariosTest(SetUpMixin):

    def test_enlace_puede_crear_alumno(self):
        self.client.login(username='enlace_t', password='Test1234!')
        tipo_alumno = TipoUsuario.objects.get(id_tipo=6)
        r = self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'crear', 'usuario': 'nuevo_alumno',
            'email': 'alumno2@test.mx', 'nombre': 'Nuevo',
            'aPaterno': 'Alumno', 'password': 'Pass1234!',
            'tipo_usuario': tipo_alumno.pk
        })
        self.assertRedirects(r, reverse('escuelas-gestionar-usuarios'))
        self.assertTrue(Usuario.objects.filter(usuario='nuevo_alumno').exists())

    def test_alumno_no_puede_gestionar_usuarios(self):
        self.client.login(username='alumno_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-gestionar-usuarios'))
        self.assertEqual(r.status_code, 403)

    def test_enlace_puede_desactivar_alumno(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-gestionar-usuarios'), {
            'action': 'desactivar', 'user_id': self.alumno.pk
        })
        self.assertRedirects(r, reverse('escuelas-gestionar-usuarios'))
        self.alumno.refresh_from_db()
        self.assertFalse(self.alumno.is_active)


# ── Registrar escuela (solo superusuario) ────────────────────────────────────

class RegistrarEscuelaTest(SetUpMixin):

    def test_superusuario_puede_registrar_escuela(self):
        self.client.login(username='admin_t', password='Test1234!')
        orizaba = Municipio.objects.get(cvegeo='30118')
        r = self.client.post(reverse('escuelas-registrar'), {
            'nombre': 'Nueva Escuela', 'nivel': 'preparatoria', 'sostenimiento': 'publico', 'turno': 'matutino',
            'clave_centro_trabajo': '30EPR0001X', 'pais': 'mexico',
            'estado_catalogo': orizaba.estado_id, 'municipio_catalogo': orizaba.id,
            'latitud': '18.850000', 'longitud': '-97.100000',
        })
        escuela = Escuela.objects.get(nombre='Nueva Escuela')
        self.assertRedirects(r, reverse('escuelas-detalle', args=[escuela.pk]))
        self.assertEqual((escuela.estado, escuela.municipio), ('Veracruz de Ignacio de la Llave', 'Orizaba'))

    def test_enlace_no_puede_registrar_escuela(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertEqual(r.status_code, 403)


# ── Editar y eliminar Enlace ──────────────────────────────────────────────────

class EditarEliminarEnlaceTest(SetUpMixin):

    def test_superusuario_puede_editar_enlace(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-editar-enlace', args=[self.enlace.pk]), {
            'nombre': 'Modificado', 'aPaterno': 'Enlace',
            'email': 'enlace_t@test.mx', 'escuela_id': self.escuela.pk, 'password': ''
        })
        self.assertRedirects(r, reverse('escuelas'))
        self.enlace.refresh_from_db()
        self.assertEqual(self.enlace.nombre, 'Modificado')

    def test_superusuario_puede_eliminar_enlace(self):
        self.client.login(username='admin_t', password='Test1234!')
        uid = self.enlace.pk
        r = self.client.post(reverse('escuelas-eliminar-enlace', args=[uid]))
        self.assertRedirects(r, reverse('escuelas'))
        self.assertFalse(Usuario.objects.filter(pk=uid).exists())

    def test_enlace_no_puede_eliminar_enlace(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-eliminar-enlace', args=[self.enlace.pk]))
        self.assertEqual(r.status_code, 403)
