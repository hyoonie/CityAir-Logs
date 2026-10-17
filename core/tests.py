import shutil
import tempfile
from datetime import timedelta

from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from django.test import TestCase, Client, SimpleTestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from core.models import (
    Usuario, TipoUsuario, Escuela, Dispositivo, DocumentoEscuela, Estado, Municipio, SincronizacionCatalogo,
)
from core.validators import (
    dividir_apellidos, generar_usuario_enlace, normalizar_telefono,
    validar_cct_mexico, validar_correo_dominio,
)


def pdf_falso(nombre='convenio.pdf', contenido=b'%PDF-1.4\n%prueba\n'):
    return SimpleUploadedFile(nombre, contenido, content_type='application/pdf')


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
            estado='Veracruz', nivel='superior', activa=True
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

def datos_enlace(escuela, **extra):
    datos = {
        'escuela': escuela.pk, 'nombre': 'María Fernanda', 'apellidos': 'Hernández Cruz',
        'correo': 'MF.Hernandez@escuela.edu.mx', 'telefono': '228 123 4567',
        'cargo': 'coordinador', 'acceso': 'temporal',
    }
    datos.update(extra)
    return datos


class CrearEnlaceTest(SetUpMixin):

    def setUp(self):
        super().setUp()
        # self.escuela ya tiene enlace (enlace_t); esta no
        self.sin_enlace = Escuela.objects.create(
            nombre='Escuela de Prueba', municipio='Xalapa', estado='Veracruz', nivel='superior',
            clave_centro_trabajo='30EPR0001X', pais='mexico', sostenimiento='publico', turno='matutino',
            latitud='19.543800', longitud='-96.910200',
        )
        self.client.login(username='admin_t', password='Test1234!')

    def test_superusuario_puede_crear_enlace(self):
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace))
        self.assertRedirects(r, reverse('escuelas-credenciales-enlace'), fetch_redirect_response=False)
        nuevo = Usuario.objects.get(email='mf.hernandez@escuela.edu.mx')
        self.assertEqual(nuevo.usuario, f'maria_{self.sin_enlace.id}')
        self.assertEqual((nuevo.nombre, nuevo.aPaterno, nuevo.aMaterno), ('María Fernanda', 'Hernández', 'Cruz'))
        self.assertEqual(nuevo.telefono, '228-123-4567')
        self.assertEqual(nuevo.cargo, 'coordinador')
        self.assertEqual(nuevo.get_cargo_display(), 'Coordinador(a)')
        self.assertEqual(nuevo.tipousuario_id, 4)
        self.assertEqual(nuevo.escuela, self.sin_enlace)
        self.assertTrue(nuevo.debe_cambiar_password)

    def test_credenciales_se_muestran_una_sola_vez_y_permiten_login(self):
        self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace))
        password = self.client.session['credenciales_enlace']['password']
        r = self.client.get(reverse('escuelas-credenciales-enlace'))
        self.assertContains(r, password)
        self.assertContains(r, 'mf.hernandez@escuela.edu.mx')
        self.assertIn('no-cache', r['Cache-Control'])
        r = self.client.get(reverse('escuelas-credenciales-enlace'))
        self.assertRedirects(r, reverse('escuelas'), fetch_redirect_response=False)
        # Acceso con correo + contraseña temporal
        self.client.logout()
        self.assertTrue(self.client.login(username='mf.hernandez@escuela.edu.mx', password=password))

    def test_get_preselecciona_escuela_y_muestra_resumen(self):
        r = self.client.get(reverse('escuelas-crear-enlace') + f'?escuela={self.sin_enlace.id}')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(str(r.context['form']['escuela'].value()), str(self.sin_enlace.id))
        self.assertContains(r, '30EPR0001X')
        self.assertContains(r, 'Próximamente')
        self.assertContains(r, 'Registrar enlace')

    def test_solo_lista_escuelas_sin_enlace(self):
        r = self.client.get(reverse('escuelas-crear-enlace'))
        escuelas = list(r.context['form'].fields['escuela'].queryset)
        self.assertIn(self.sin_enlace, escuelas)
        self.assertNotIn(self.escuela, escuelas)
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.escuela))
        self.assertIn('escuela', r.context['form'].errors)
        self.assertFalse(Usuario.objects.filter(email='mf.hernandez@escuela.edu.mx').exists())

    def test_validaciones(self):
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(
            self.sin_enlace, correo='enlace@hotmail.com', telefono='123', nombre='', cargo='', apellidos=''))
        errores = r.context['form'].errors
        for campo in ('correo', 'telefono', 'nombre', 'cargo', 'apellidos'):
            self.assertIn(campo, errores)

    def test_correo_duplicado(self):
        ocupado = crear_usuario('ocupado', 3)
        ocupado.email = 'mf.hernandez@escuela.edu.mx'
        ocupado.save()
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace))
        self.assertIn('correo', r.context['form'].errors)

    def test_invitacion_por_correo_no_disponible(self):
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace, acceso='invitacion'))
        self.assertIn('acceso', r.context['form'].errors)

    def test_usuario_autogenerado_sin_colision(self):
        crear_usuario(f'maria_{self.sin_enlace.id}', 3)
        self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(self.sin_enlace))
        nuevo = Usuario.objects.get(email='mf.hernandez@escuela.edu.mx')
        self.assertEqual(nuevo.usuario, f'maria_{self.sin_enlace.id}_2')

    def test_enlace_no_puede_crear_enlace(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-crear-enlace'))
        self.assertEqual(r.status_code, 403)

    def test_alumno_no_puede_crear_enlace(self):
        self.client.login(username='alumno_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-crear-enlace'))
        self.assertEqual(r.status_code, 403)


# ── Gestionar usuarios (Enlace) ───────────────────────────────────────────────

class GestionarUsuariosTest(SetUpMixin):

    def test_enlace_puede_crear_alumno(self):
        """Tras la refactorización, el alumno individual se crea vía la API JSON."""
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.post(
            reverse('api-escuelas-alumno-crear'),
            data=json.dumps({
                'nombre': 'Nuevo',
                'apellidos': 'Alumno Prueba',
                'correo': 'nuevo.alumno@escuela.edu.mx',
                'telefono': '',
            }),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.content)
        self.assertIn('creado', data)
        u = Usuario.objects.get(email='nuevo.alumno@escuela.edu.mx')
        self.assertEqual(u.tipousuario_id, 6)
        self.assertEqual(u.escuela, self.escuela)
        self.assertTrue(u.debe_cambiar_password)


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


# ── Validadores ───────────────────────────────────────────────────────────────

class ValidadoresTest(SimpleTestCase):

    def test_cct_valido(self):
        for cct in ('30EPR0001X', '30EPR2562L', '09DPR1234A', '32DES0001Z'):
            validar_cct_mexico(cct)

    def test_cct_invalido(self):
        for cct in ('NE001', '33EPR0001X', '00EPR0001X', '30EP0001X', '30EPR00011', '30epr0001x', '30EPR0001XX'):
            with self.assertRaises(ValidationError, msg=cct):
                validar_cct_mexico(cct)

    def test_telefono_normalizado(self):
        self.assertEqual(normalizar_telefono('228 165 0525'), '228-165-0525')
        self.assertEqual(normalizar_telefono('228-165-0525'), '228-165-0525')
        self.assertEqual(normalizar_telefono('2281650525'), '228-165-0525')

    def test_telefono_invalido(self):
        for tel in ('228 165 052', '(228) 165 0525', '228.165.0525', 'abc-def-ghij', '228 165 05255'):
            with self.assertRaises(ValidationError, msg=tel):
                normalizar_telefono(tel)

    def test_dominios_correo(self):
        for correo in ('a@gmail.com', 'b@edu.mx', 'c@itx.edu.mx', 'd@uv.edu.mx'):
            validar_correo_dominio(correo)
        for correo in ('a@hotmail.com', 'b@edu.mx.com', 'c@fakeedu.mx', 'd@x.gmail.com'):
            with self.assertRaises(ValidationError, msg=correo):
                validar_correo_dominio(correo)

    def test_usuario_enlace_autogenerado(self):
        self.assertEqual(generar_usuario_enlace('María Fernanda', 12, lambda u: False), 'maria_12')
        ocupados = {'maria_12'}
        self.assertEqual(generar_usuario_enlace('María', 12, lambda u: u in ocupados), 'maria_12_2')
        self.assertLessEqual(len(generar_usuario_enlace('x' * 80, 123456, lambda u: False)), 45)

    def test_dividir_apellidos(self):
        self.assertEqual(dividir_apellidos('Hernández Cruz'), ('Hernández', 'Cruz'))
        self.assertEqual(dividir_apellidos('Hernández'), ('Hernández', ''))
        self.assertEqual(dividir_apellidos('de la Cruz López'), ('de', 'la Cruz López'))


# ── Registrar / editar escuela ───────────────────────────────────────────────

class ConveniosTmpMixin:
    """Guarda los PDF de prueba en una carpeta temporal en lugar de private_media/."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp_convenios = tempfile.mkdtemp()
        campo = DocumentoEscuela._meta.get_field('archivo')
        cls._storage_original = campo.storage
        campo.storage = FileSystemStorage(location=cls._tmp_convenios)

    @classmethod
    def tearDownClass(cls):
        DocumentoEscuela._meta.get_field('archivo').storage = cls._storage_original
        shutil.rmtree(cls._tmp_convenios, ignore_errors=True)
        super().tearDownClass()


def veracruz_xalapa():
    return Estado.objects.get(cve_ent='30'), Municipio.objects.get(cvegeo='30087')


def datos_escuela(**extra):
    estado, municipio = veracruz_xalapa()
    datos = {
        'nombre': 'Escuela de Prueba', 'clave_centro_trabajo': '30epr0001x',
        'nivel': 'superior', 'sostenimiento': 'publico', 'turno': 'matutino',
        'telefono': '228 165 0525', 'correo_institucional': 'contacto@escuela.edu.mx',
        'pais': 'mexico', 'estado_catalogo': str(estado.id), 'municipio_catalogo': str(municipio.id),
        'colonia': 'Centro', 'calle': 'Av. Ejemplo', 'numero': '100', 'codigo_postal': '91000',
        'latitud': '19.543800', 'longitud': '-96.910200',
        'tipo': 'convenio', 'fecha_firma': (timezone.localdate() - timedelta(days=3)).isoformat(),
    }
    datos.update(extra)
    return datos


class RegistrarEscuelaTest(ConveniosTmpMixin, SetUpMixin):

    def registrar(self, archivo=None, **extra):
        datos = datos_escuela(**extra)
        datos['archivo'] = archivo if archivo is not None else pdf_falso()
        return self.client.post(reverse('escuelas-registrar'), datos)

    def test_superusuario_puede_registrar_escuela(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.registrar()
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        self.assertRedirects(r, f"{reverse('escuelas-crear-enlace')}?escuela={escuela.id}", fetch_redirect_response=False)
        # Normalizaciones
        self.assertEqual(escuela.clave_centro_trabajo, '30EPR0001X')
        self.assertEqual(escuela.telefono, '228-165-0525')
        self.assertTrue(escuela.activa)
        self.assertIsNotNone(escuela.fecha_registro)
        convenio = escuela.convenio
        self.assertEqual(convenio.tipo, 'convenio')
        self.assertEqual(convenio.subido_por, self.admin)
        self.assertTrue(convenio.archivo.storage.exists(convenio.archivo.name))

    def test_get_muestra_formulario(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'Identificación')
        self.assertContains(r, 'Elegir en el mapa')
        self.assertContains(r, 'Guardar escuela')

    def test_obligatorios(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-registrar'), {})
        self.assertEqual(r.status_code, 200)
        for campo in ('nombre', 'nivel', 'sostenimiento', 'pais', 'latitud', 'longitud'):
            self.assertIn(campo, r.context['escuela_form'].errors)
        for campo in ('archivo', 'fecha_firma'):
            self.assertIn(campo, r.context['convenio_form'].errors)
        self.assertFalse(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_opcionales_vacios(self):
        self.client.login(username='admin_t', password='Test1234!')
        self.registrar(clave_centro_trabajo='', turno='', telefono='', correo_institucional='',
                       colonia='', calle='', numero='', codigo_postal='')
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        self.assertIsNone(escuela.clave_centro_trabajo)
        self.assertIsNone(escuela.telefono)

    def test_validaciones_de_formato(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.registrar(clave_centro_trabajo='NE001', telefono='12345',
                           correo_institucional='contacto@hotmail.com', codigo_postal='91A00',
                           latitud='95', nivel='primaria')
        errores = r.context['escuela_form'].errors
        for campo in ('clave_centro_trabajo', 'telefono', 'correo_institucional', 'codigo_postal', 'latitud', 'nivel'):
            self.assertIn(campo, errores)
        self.assertFalse(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_cct_solo_se_valida_en_mexico(self):
        self.client.login(username='admin_t', password='Test1234!')
        self.registrar(pais='colombia', clave_centro_trabajo='111001000001', estado_catalogo='', municipio_catalogo='',
                       estado='Antioquia', municipio='Medellín')
        self.assertTrue(Escuela.objects.filter(nombre='Escuela de Prueba', pais='colombia').exists())

    def test_cct_unica_por_pais(self):
        self.client.login(username='admin_t', password='Test1234!')
        self.registrar()
        r = self.registrar(nombre='Otra escuela')
        self.assertEqual(r.status_code, 200)
        self.assertFalse(Escuela.objects.filter(nombre='Otra escuela').exists())
        self.assertIn('Ya existe una escuela con esa clave CCT', str(r.context['escuela_form'].errors))

    def test_fecha_firma_futura(self):
        self.client.login(username='admin_t', password='Test1234!')
        manana = (timezone.localdate() + timedelta(days=1)).isoformat()
        r = self.registrar(fecha_firma=manana)
        self.assertIn('fecha_firma', r.context['convenio_form'].errors)
        hoy = timezone.localdate().isoformat()
        self.registrar(fecha_firma=hoy)
        self.assertTrue(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_pdf_invalido(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.registrar(archivo=SimpleUploadedFile('convenio.txt', b'hola', content_type='text/plain'))
        self.assertIn('archivo', r.context['convenio_form'].errors)
        r = self.registrar(archivo=pdf_falso(contenido=b'no es un pdf'))
        self.assertIn('archivo', r.context['convenio_form'].errors)
        r = self.registrar(archivo=pdf_falso(contenido=b'%PDF-' + b'0' * (5 * 1024 * 1024)))
        self.assertIn('archivo', r.context['convenio_form'].errors)
        self.assertFalse(Escuela.objects.filter(nombre='Escuela de Prueba').exists())

    def test_staff_no_superusuario_va_al_detalle(self):
        staff = crear_usuario('staff_t', 1)
        staff.is_staff = True
        staff.save()
        self.client.login(username='staff_t', password='Test1234!')
        r = self.registrar()
        escuela = Escuela.objects.get(nombre='Escuela de Prueba')
        self.assertRedirects(r, reverse('escuelas-detalle', args=[escuela.id]), fetch_redirect_response=False)

    def test_enlace_no_puede_registrar_escuela(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-registrar'))
        self.assertEqual(r.status_code, 403)


class EditarEscuelaTest(ConveniosTmpMixin, SetUpMixin):

    def setUp(self):
        super().setUp()
        self.client.login(username='admin_t', password='Test1234!')
        datos = datos_escuela()
        datos['archivo'] = pdf_falso()
        self.client.post(reverse('escuelas-registrar'), datos)
        self.nueva = Escuela.objects.get(nombre='Escuela de Prueba')

    def test_editar_sin_reemplazar_pdf(self):
        archivo_antes = self.nueva.convenio.archivo.name
        r = self.client.post(reverse('escuelas-editar', args=[self.nueva.id]),
                             datos_escuela(nombre='Escuela Renombrada'))
        self.assertRedirects(r, reverse('escuelas-detalle', args=[self.nueva.id]), fetch_redirect_response=False)
        self.nueva.refresh_from_db()
        self.assertEqual(self.nueva.nombre, 'Escuela Renombrada')
        self.assertEqual(self.nueva.convenio.archivo.name, archivo_antes)

    def test_reemplazar_pdf_borra_el_anterior(self):
        convenio = self.nueva.convenio
        anterior = convenio.archivo.name
        datos = datos_escuela()
        datos['archivo'] = pdf_falso('nuevo.pdf')
        self.client.post(reverse('escuelas-editar', args=[self.nueva.id]), datos)
        convenio.refresh_from_db()
        self.assertNotEqual(convenio.archivo.name, anterior)
        self.assertFalse(convenio.archivo.storage.exists(anterior))
        self.assertTrue(convenio.archivo.storage.exists(convenio.archivo.name))
        self.assertEqual(DocumentoEscuela.objects.filter(escuela=self.nueva).count(), 1)

    def test_escuela_sin_convenio_exige_pdf(self):
        # Escuela registrada antes de esta versión (sin convenio ni datos nuevos)
        r = self.client.post(reverse('escuelas-editar', args=[self.escuela.id]), datos_escuela(
            nombre='Escuela Test', clave_centro_trabajo=''))
        self.assertIn('archivo', r.context['convenio_form'].errors)

    def test_enlace_no_puede_editar(self):
        self.client.login(username='enlace_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-editar', args=[self.nueva.id]))
        self.assertEqual(r.status_code, 403)

    def test_detalle_muestra_datos_nuevos(self):
        r = self.client.get(reverse('escuelas-detalle', args=[self.nueva.id]))
        self.assertContains(r, '30EPR0001X')
        self.assertContains(r, 'Público')
        self.assertContains(r, 'Ver convenio')
        self.assertContains(r, 'Editar escuela')


class DescargaConvenioTest(ConveniosTmpMixin, SetUpMixin):

    def setUp(self):
        super().setUp()
        DocumentoEscuela.objects.create(
            escuela=self.escuela, tipo='convenio', archivo=pdf_falso(),
            fecha_firma=timezone.localdate(),
        )
        self.otra = Escuela.objects.create(nombre='Otra', municipio='Orizaba', estado='Veracruz', nivel='secundaria')
        self.enlace_otra = crear_usuario('enlace_otra', 4, escuela=self.otra)

    def descargar(self, usuario):
        self.client.login(username=usuario, password='Test1234!')
        return self.client.get(reverse('escuelas-convenio', args=[self.escuela.id]))

    def test_admin_y_enlace_propio_descargan(self):
        for usuario in ('admin_t', 'enlace_t'):
            r = self.descargar(usuario)
            self.assertEqual(r.status_code, 200, usuario)
            self.assertEqual(r['Content-Type'], 'application/pdf')
            self.assertTrue(b''.join(r.streaming_content).startswith(b'%PDF-'))

    def test_otros_no_descargan(self):
        for usuario in ('enlace_otra', 'docente_t', 'alumno_t', 'general_t'):
            self.assertEqual(self.descargar(usuario).status_code, 403, usuario)

    def test_sin_convenio_404(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-convenio', args=[self.otra.id]))
        self.assertEqual(r.status_code, 404)


# ── Editar y eliminar Enlace ──────────────────────────────────────────────────

class EditarEliminarEnlaceTest(SetUpMixin):

    def test_superusuario_puede_editar_enlace(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-editar-enlace', args=[self.enlace.pk]), datos_enlace(
            self.escuela, nombre='Modificado', apellidos='Pérez de la Luz', correo='enlace@itx.edu.mx'))
        self.assertRedirects(r, reverse('escuelas'))
        self.enlace.refresh_from_db()
        self.assertEqual(self.enlace.nombre, 'Modificado')
        self.assertEqual((self.enlace.aPaterno, self.enlace.aMaterno), ('Pérez', 'de la Luz'))
        self.assertEqual(self.enlace.email, 'enlace@itx.edu.mx')
        self.assertEqual(self.enlace.telefono, '228-123-4567')
        self.assertFalse(self.enlace.debe_cambiar_password)

    def test_get_editar_precarga_datos(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.get(reverse('escuelas-editar-enlace', args=[self.enlace.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context['form']['apellidos'].value(), 'User')
        self.assertIn(self.escuela, list(r.context['form'].fields['escuela'].queryset))

    def test_restablecer_password_temporal(self):
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-editar-enlace', args=[self.enlace.pk]), datos_enlace(
            self.escuela, correo='enlace@itx.edu.mx', restablecer_password='on'))
        self.assertRedirects(r, reverse('escuelas-credenciales-enlace'), fetch_redirect_response=False)
        password = self.client.session['credenciales_enlace']['password']
        self.enlace.refresh_from_db()
        self.assertTrue(self.enlace.debe_cambiar_password)
        self.assertTrue(self.enlace.check_password(password))

    def test_no_puede_moverse_a_escuela_con_enlace(self):
        otra = Escuela.objects.create(nombre='Otra', municipio='Orizaba', estado='Veracruz', nivel='secundaria')
        crear_usuario('enlace_otra', 4, escuela=otra)
        self.client.login(username='admin_t', password='Test1234!')
        r = self.client.post(reverse('escuelas-editar-enlace', args=[self.enlace.pk]), datos_enlace(
            otra, correo='enlace@itx.edu.mx'))
        self.assertIn('escuela', r.context['form'].errors)
        self.enlace.refresh_from_db()
        self.assertEqual(self.enlace.escuela, self.escuela)

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


# ── Cambio forzado de contraseña temporal ─────────────────────────────────────

FORZAR_CAMBIO = 'core.middleware.ForzarCambioPasswordMiddleware'


# settings.py no está versionado: la prueba registra el middleware por su cuenta
@override_settings(MIDDLEWARE=[m for m in settings.MIDDLEWARE if m != FORZAR_CAMBIO] + [FORZAR_CAMBIO])
class CambioPasswordForzadoTest(SetUpMixin):

    def setUp(self):
        super().setUp()
        self.enlace.set_password('Temporal123')
        self.enlace.debe_cambiar_password = True
        self.enlace.save()

    def test_login_redirige_a_cambiar_password(self):
        r = self.client.post(reverse('login'), {'username': 'enlace_t@test.mx', 'password': 'Temporal123'})
        self.assertRedirects(r, reverse('configuracion', args=[self.enlace.id]), fetch_redirect_response=False)

    def test_middleware_bloquea_otras_paginas(self):
        self.client.login(username='enlace_t', password='Temporal123')
        r = self.client.get(reverse('escuelas'))
        self.assertRedirects(r, reverse('configuracion', args=[self.enlace.id]), fetch_redirect_response=False)
        r = self.client.get(reverse('configuracion', args=[self.enlace.id]))
        self.assertEqual(r.status_code, 200)

    def test_al_cambiar_password_se_libera(self):
        self.client.login(username='enlace_t', password='Temporal123')
        self.client.post(reverse('cambiar-password'), {
            'old_password': 'Temporal123', 'new_password1': 'NuevaSegura#2026', 'new_password2': 'NuevaSegura#2026',
        })
        self.enlace.refresh_from_db()
        self.assertFalse(self.enlace.debe_cambiar_password)
        r = self.client.get(reverse('escuelas'))
        self.assertEqual(r.status_code, 200)

    def test_password_debil_muestra_error_visible_en_espanol(self):
        self.client.login(username='enlace_t', password='Temporal123')
        r = self.client.post(reverse('cambiar-password'), {
            'old_password': 'Temporal123', 'new_password1': 'abc123', 'new_password2': 'abc123',
        }, follow=True)
        self.assertContains(r, 'alert-danger')
        self.assertNotContains(r, 'alert-error')
        self.assertContains(r, 'demasiado corta')
        # Los requisitos se muestran antes de intentar
        self.assertContains(r, '8 caracteres')
        self.enlace.refresh_from_db()
        self.assertTrue(self.enlace.debe_cambiar_password)

    def test_usuario_sin_bandera_no_se_bloquea(self):
        self.client.login(username='docente_t', password='Test1234!')
        self.assertEqual(self.client.get(reverse('escuelas')).status_code, 200)


# ── Errores de formulario en español ──────────────────────────────────────────

INGLES = ('This field is required', 'Enter a valid', 'Select a valid choice', 'Ensure this value',
          'already exists', 'Enter a number', 'Ensure that there are')


class ErroresEnEspanolTest(ConveniosTmpMixin, SetUpMixin):

    def setUp(self):
        super().setUp()
        self.client.login(username='admin_t', password='Test1234!')

    def assert_espanol(self, respuesta):
        html = respuesta.content.decode()
        for frase in INGLES:
            self.assertNotIn(frase, html, frase)

    def test_registrar_escuela_vacio(self):
        r = self.client.post(reverse('escuelas-registrar'), {})
        self.assertContains(r, 'Este campo es requerido')
        self.assert_espanol(r)

    def test_registrar_escuela_valores_invalidos(self):
        datos = datos_escuela(nivel='primaria', latitud='abc', longitud='9999999', correo_institucional='x',
                              nombre='n' * 200, fecha_firma='no-es-fecha')
        datos['archivo'] = pdf_falso()
        r = self.client.post(reverse('escuelas-registrar'), datos)
        self.assertContains(r, 'Escoja una opción válida')
        self.assertContains(r, 'Introduzca un número')
        self.assert_espanol(r)

    def test_editar_escuela_vacio(self):
        r = self.client.post(reverse('escuelas-editar', args=[self.escuela.id]), {})
        self.assertContains(r, 'Este campo es requerido')
        self.assert_espanol(r)

    def test_registrar_enlace_vacio_y_correo_invalido(self):
        sin_enlace = Escuela.objects.create(nombre='Sin enlace', municipio='X', estado='Y', nivel='superior')
        r = self.client.post(reverse('escuelas-crear-enlace'), {})
        self.assertContains(r, 'Este campo es requerido')
        self.assert_espanol(r)
        r = self.client.post(reverse('escuelas-crear-enlace'), datos_enlace(sin_enlace, correo='no-es-correo'))
        self.assertContains(r, 'Introduzca una dirección de correo electrónico válida')
        self.assert_espanol(r)

    def test_editar_enlace_vacio(self):
        r = self.client.post(reverse('escuelas-editar-enlace', args=[self.enlace.pk]), {})
        self.assertContains(r, 'Este campo es requerido')
        self.assert_espanol(r)
