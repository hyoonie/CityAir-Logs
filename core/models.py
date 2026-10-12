import uuid

from django.db import models
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin
)
from django.utils import timezone

from .storage import get_convenios_storage

# Catálogo cerrado de México (INEGI AGEEML). Se carga con una migración y se actualiza con
# `python manage.py sincronizar_catalogo_geo`. Nunca se borran filas: las bajas quedan activo=False.
class Estado(models.Model):
    cve_ent = models.CharField(max_length=2, unique=True)
    nombre = models.CharField(max_length=100)
    nombre_abreviado = models.CharField(max_length=20, blank=True, default='')
    activo = models.BooleanField(default=True)

    class Meta:
        db_table = 'catalogo_estados'
        ordering = ['cve_ent']

    def __str__(self):
        return self.nombre


class Municipio(models.Model):
    cvegeo = models.CharField(max_length=5, unique=True)
    estado = models.ForeignKey(Estado, on_delete=models.PROTECT, related_name='municipios')
    cve_mun = models.CharField(max_length=3)
    nombre = models.CharField(max_length=150)
    activo = models.BooleanField(default=True)

    class Meta:
        db_table = 'catalogo_municipios'
        ordering = ['nombre']

    def __str__(self):
        return f"{self.nombre}, {self.estado.nombre}"


class SincronizacionCatalogo(models.Model):
    fecha = models.DateTimeField(default=timezone.now)
    exitosa = models.BooleanField(default=True)
    resumen = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default='')

    class Meta:
        db_table = 'catalogo_sincronizaciones'
        ordering = ['-fecha']

    def __str__(self):
        return f"{self.fecha:%Y-%m-%d %H:%M} - {'OK' if self.exitosa else 'ERROR'}"


# Modelo/tabla: Escuela
class Escuela(models.Model):
    NIVEL_CHOICES = [
        ('secundaria', 'Secundaria'),
        ('media_superior', 'Media superior'),
        ('superior', 'Superior'),
    ]
    SOSTENIMIENTO_CHOICES = [
        ('publico', 'Público'),
        ('privado', 'Privado'),
    ]
    TURNO_CHOICES = [
        ('matutino', 'Matutino'),
        ('vespertino', 'Vespertino'),
        ('mixto', 'Mixto'),
    ]
    PAIS_CHOICES = [
        ('mexico', 'México'),
        ('colombia', 'Colombia'),
    ]

    # Identificación
    nombre = models.CharField(max_length=128)
    clave_centro_trabajo = models.CharField(max_length=20, null=True, blank=True)
    nivel = models.CharField(max_length=20, choices=NIVEL_CHOICES)
    sostenimiento = models.CharField(max_length=10, choices=SOSTENIMIENTO_CHOICES, null=True, blank=True)
    turno = models.CharField(max_length=10, choices=TURNO_CHOICES, null=True, blank=True)
    telefono = models.CharField(max_length=12, null=True, blank=True)
    correo_institucional = models.EmailField(max_length=255, null=True, blank=True)
    fecha_registro = models.DateTimeField(default=timezone.now, editable=False)

    # Ubicación
    pais = models.CharField(max_length=20, choices=PAIS_CHOICES, null=True, blank=True)
    # Para México se eligen del catálogo cerrado; estado y municipio (texto) guardan una copia
    # del nombre y son los únicos que se usan fuera de México.
    estado_catalogo = models.ForeignKey(Estado, on_delete=models.PROTECT, null=True, blank=True, related_name='escuelas')
    municipio_catalogo = models.ForeignKey(Municipio, on_delete=models.PROTECT, null=True, blank=True, related_name='escuelas')
    estado = models.CharField(max_length=64)
    municipio = models.CharField(max_length=150)
    colonia = models.CharField(max_length=128, null=True, blank=True)
    calle = models.CharField(max_length=100, null=True, blank=True)
    numero = models.CharField(max_length=10, null=True, blank=True)
    codigo_postal = models.CharField(max_length=8, null=True, blank=True)
    latitud = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitud = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)

    activa = models.BooleanField(default=True)

    class Meta:
        db_table = 'escuelas'
        constraints = [
            # MySQL trata los NULL como distintos: la CCT vacía se guarda como NULL
            models.UniqueConstraint(
                fields=['pais', 'clave_centro_trabajo'],
                name='uniq_escuela_pais_cct',
                violation_error_message='Ya existe una escuela con esa clave CCT en este país.',
            ),
        ]

    def __str__(self):
        return f"{self.nombre} ({self.municipio}, {self.estado})"

    @property
    def convenio(self):
        return self.documentos.filter(tipo='convenio').first()

    @property
    def enlace(self):
        return self.usuarios.filter(tipousuario__id_tipo=4).first()


def ruta_documento_escuela(instance, filename):
    return f"escuela_{instance.escuela_id}/{instance.tipo}_{uuid.uuid4().hex}.pdf"


# Modelo/tabla: Documento de autorización de la escuela (convenio)
class DocumentoEscuela(models.Model):
    TIPO_CHOICES = [
        ('convenio', 'Convenio'),
    ]

    escuela = models.ForeignKey(Escuela, on_delete=models.CASCADE, related_name='documentos')
    tipo = models.CharField(max_length=20, choices=TIPO_CHOICES, default='convenio')
    archivo = models.FileField(upload_to=ruta_documento_escuela, storage=get_convenios_storage, max_length=255)
    fecha_firma = models.DateField()
    fecha_carga = models.DateTimeField(auto_now=True)
    subido_por = models.ForeignKey('Usuario', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')

    class Meta:
        db_table = 'documentos_escuela'
        constraints = [
            models.UniqueConstraint(fields=['escuela', 'tipo'], name='uniq_documento_escuela_tipo'),
        ]

    def __str__(self):
        return f"{self.get_tipo_display()} - {self.escuela.nombre}"


# Modelo/tabla: Tipo de Usuario
class TipoUsuario(models.Model):
    id_tipo = models.IntegerField(primary_key=True)
    nombre_tipo = models.CharField(max_length=45)
    descripcion = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'tipousuario'

    def __str__(self):
        return self.nombre_tipo 

# Modelo/tabla: Enfermedad
class Enfermedad(models.Model):
    idEnfermedad = models.IntegerField(primary_key=True)
    tipo = models.CharField(max_length=45)
    descripcion = models.CharField(max_length=45)

    class Meta:
        db_table = 'enfermedad'

    def __str__(self):
        return self.tipo

# Modelo/tabla: dispositivo
class Dispositivo(models.Model):
    idDispositivo = models.CharField(primary_key=True, max_length=64)
    dev_eui = models.CharField(max_length=100, unique=True, null=True, blank=True)
    pais = models.CharField(max_length=64)
    estado = models.CharField(max_length=64)
    ciudad = models.CharField(max_length=64)
    calle = models.CharField(max_length=100)
    colonia = models.CharField(max_length=128)
    codigo_postal = models.CharField(max_length=8)
    numero = models.CharField(max_length=10)
    descripcion = models.TextField(null=True, blank=True)
    tipo_sensor = models.CharField(max_length=20, null=True, blank=True)
    escuela = models.ForeignKey('Escuela', on_delete=models.SET_NULL, null=True, blank=True, related_name='dispositivos')

    class Meta:
        db_table = 'dispositivos'

    def __str__(self):
        return f"Dispositivo {self.idDispositivo} - {self.ciudad}, {self.pais}"
    

# Gestor de usuario
class UsuarioManager(BaseUserManager):

    # Creacion de usuario normal
    def create_user(self, usuario, email, password=None, **extra_fields):
        if not usuario:
            raise ValueError('El campo "usuario" es obligatorio.')
        if not email:
            raise ValueError('El campo "email" es obligatorio.')
        
        email = self.normalize_email(email)
        user = self.model(usuario=usuario, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        
        return user

    # Creacion de superusuario
    def create_superuser(self, usuario, email, password, **extra_fields):
        user = self.create_user(
            usuario,
            email,
            password,
            **extra_fields 
        )
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)

        return user

# Modelo/tabla: usuario
class Usuario(AbstractBaseUser, PermissionsMixin):

    usuario = models.CharField(max_length=45, unique=True)
    email = models.EmailField(max_length=255, unique=True)

    nombre = models.CharField(max_length=45)
    aPaterno = models.CharField(max_length=45)
    aMaterno = models.CharField(max_length=45)
    edad = models.IntegerField(null=True, blank=True)
    atleta = models.BooleanField(default=False)
    telefono = models.CharField(max_length=12, null=True, blank=True)
    CARGO_CHOICES = [
        ('director', 'Director(a)'),
        ('subdirector', 'Subdirector(a)'),
        ('coordinador', 'Coordinador(a)'),
    ]
    cargo = models.CharField(max_length=100, choices=CARGO_CHOICES, null=True, blank=True)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    debe_cambiar_password = models.BooleanField(default=False)

    tipousuario = models.ForeignKey(TipoUsuario, on_delete=models.SET_NULL, null=True, blank=True)
    enfermedad = models.ForeignKey(Enfermedad, on_delete=models.SET_NULL, null=True, blank=True)
    escuela = models.ForeignKey(Escuela, on_delete=models.SET_NULL, null=True, blank=True, related_name='usuarios')
    
    objects = UsuarioManager()

    USERNAME_FIELD = 'usuario'
    
    REQUIRED_FIELDS = ['email', 'nombre', 'aPaterno']

    def __str__(self):
        return self.usuario