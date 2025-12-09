from django.db import models
from django.contrib.auth.models import (
    AbstractBaseUser,
    BaseUserManager,
    PermissionsMixin
)

# Modelo/tabla: Tipo de Usuario
class TipoUsuario(models.Model):
    # CORREGIDO: Usando 'id_tipo' como pediste
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
    idDispositivo = models.IntegerField(primary_key=True)
    dev_eui = models.CharField(max_length=100, unique=True, null=True, blank=True)
    pais = models.CharField(max_length=45)
    estado = models.CharField(max_length=45)
    ciudad = models.CharField(max_length=45)
    calleynumero = models.CharField(max_length=45)

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

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    tipousuario = models.ForeignKey(TipoUsuario, on_delete=models.SET_NULL, null=True, blank=True)
    enfermedad = models.ForeignKey(Enfermedad, on_delete=models.SET_NULL, null=True, blank=True)
    
    objects = UsuarioManager()

    USERNAME_FIELD = 'usuario'
    
    REQUIRED_FIELDS = ['email', 'nombre', 'aPaterno']

    def __str__(self):
        return self.usuario