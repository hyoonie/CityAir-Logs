"""
core/usuarios_escuela.py
Lógica de dominio para el registro de Docentes y Alumnos por el Enlace.

Tres funciones públicas principales:
  - validar_filas(filas)      → (errores, filas_limpias)   sin tocar la BD si hay errores de formato
  - crear_usuarios(...)       → lista de credenciales creadas
  - enlace_de_escuela(user)   → Escuela | None

No depende de `request`; es agnóstico a la capa HTTP.
"""

import re

from django.contrib.auth.hashers import PBKDF2PasswordHasher
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models.functions import Lower

from .models import TipoUsuario, Usuario
from .validators import (
    dividir_apellidos,
    generar_password_temporal,
    generar_usuario_enlace,
    normalizar_telefono,
    validar_correo_dominio,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

ROL_DOCENTE = 5
ROL_ALUMNO = 6
MAX_FILAS_LOTE = 200


# ──────────────────────────────────────────────────────────────────────────────
# Hash ligero para contraseñas temporales (punto 4.4 del plan)
# ──────────────────────────────────────────────────────────────────────────────

class _LightPBKDF2Hasher(PBKDF2PasswordHasher):
    """PBKDF2-SHA256 con 10 000 iteraciones para contraseñas temporales de un solo uso.

    El hash resultante (`pbkdf2_sha256$10000$…`) es compatible con el login
    normal de Django.  Al primer inicio de sesión exitoso Django lo actualiza
    automáticamente al algoritmo configurado en settings (porque las iteraciones
    difieren).  El usuario cambia la contraseña de inmediato
    (debe_cambiar_password=True), así que la contraseña temporal nunca persiste
    con el hash ligero.

    Medición local: 50 hashes ligeros ≈ 0.36 s (vs. ~34 s con 1 000 000 iter.).
    """
    iterations = 10_000


_light_hasher = _LightPBKDF2Hasher()


def _hash_ligero(password: str) -> str:
    """Devuelve el hash PBKDF2 con 10 000 iteraciones listo para guardar en BD."""
    return _light_hasher.encode(password, _light_hasher.salt())


# ──────────────────────────────────────────────────────────────────────────────
# Permiso
# ──────────────────────────────────────────────────────────────────────────────

def enlace_de_escuela(user):
    """Devuelve la Escuela si `user` es Enlace activo con escuela activa; si no, None."""
    if (
        user.is_authenticated
        and user.tipousuario_id == 4
        and user.escuela_id is not None
        and getattr(user.escuela, 'activa', False)
    ):
        return user.escuela
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Validación de una sola persona
# ──────────────────────────────────────────────────────────────────────────────

def _normalizar_texto(valor, max_largo, campo):
    """Colapsa espacios múltiples, quita extremos y valida longitud."""
    valor = re.sub(r'\s+', ' ', (valor or '').strip())
    if not valor:
        raise ValidationError({campo: 'Este campo es obligatorio.'})
    if len(valor) > max_largo:
        raise ValidationError({campo: f'Máximo {max_largo} caracteres.'})
    return valor


def validar_persona(datos: dict) -> dict:
    """Valida y normaliza una fila sin tocar la BD.

    Entrada:  {'nombre': str, 'apellidos': str, 'correo': str, 'telefono': str}
    Salida:   datos limpios {'nombre', 'aPaterno', 'aMaterno', 'correo', 'telefono'}
    Lanza:    ValidationError con dict {campo: mensaje} si hay errores.
    """
    errores = {}
    limpios = {}

    # nombre
    try:
        limpios['nombre'] = _normalizar_texto(datos.get('nombre'), 45, 'nombre')
    except ValidationError as exc:
        errores.update(exc.message_dict)

    # apellidos → aPaterno + aMaterno
    try:
        apellidos = _normalizar_texto(datos.get('apellidos'), 91, 'apellidos')
        paterno, materno = dividir_apellidos(apellidos)
        if len(paterno) > 45:
            raise ValidationError({'apellidos': 'Apellido paterno: máximo 45 caracteres.'})
        if len(materno) > 45:
            raise ValidationError({'apellidos': 'Apellido materno: máximo 45 caracteres.'})
        limpios['aPaterno'] = paterno
        limpios['aMaterno'] = materno
    except ValidationError as exc:
        errores.update(exc.message_dict)

    # correo
    correo_raw = (datos.get('correo') or '').strip().lower()
    if not correo_raw:
        errores['correo'] = 'Este campo es obligatorio.'
    elif len(correo_raw) > 255:
        errores['correo'] = 'Máximo 255 caracteres.'
    elif '@' not in correo_raw:
        errores['correo'] = 'Formato de correo inválido.'
    else:
        try:
            validar_correo_dominio(correo_raw)
            limpios['correo'] = correo_raw
        except ValidationError as exc:
            errores['correo'] = exc.message if isinstance(exc.message, str) else exc.messages[0]

    # teléfono (opcional)
    tel = (datos.get('telefono') or '').strip()
    if tel:
        try:
            limpios['telefono'] = normalizar_telefono(tel)
        except ValidationError as exc:
            errores['telefono'] = exc.message if isinstance(exc.message, str) else exc.messages[0]
    else:
        limpios['telefono'] = None

    if errores:
        raise ValidationError(errores)
    return limpios


# ──────────────────────────────────────────────────────────────────────────────
# Validación de un arreglo de filas (ajuste B del plan aprobado)
# ──────────────────────────────────────────────────────────────────────────────

def validar_filas(filas: list) -> tuple:
    """Valida un arreglo de filas en una sola pasada.

    Incluye:
      - Validación de formato por fila (llama a validar_persona internamente).
      - Detección de correos repetidos dentro del lote.
      - Consulta a la BD para correos ya existentes (una sola consulta).

    Devuelve:
      (errores, filas_limpias)
        errores:      lista de {'fila': int, 'campo': str, 'mensaje': str}
                      — lista vacía si todo está OK.
        filas_limpias: lista paralela a `filas` con los datos limpios de cada
                      fila válida, o None en las posiciones con error.
                      Solo es confiable cuando errores == [].
    """
    errores = []
    filas_limpias = []
    correos_en_lote: dict[str, int] = {}  # correo → número de fila (1-based)

    # Paso 1: validación de formato por fila
    for i, fila in enumerate(filas):
        try:
            limpios = validar_persona(fila)
            correo = limpios['correo']
            if correo in correos_en_lote:
                errores.append({
                    'fila': i + 1,
                    'campo': 'correo',
                    'mensaje': f'Correo repetido (ya aparece en la fila {correos_en_lote[correo]}).',
                })
                filas_limpias.append(None)
            else:
                correos_en_lote[correo] = i + 1
                filas_limpias.append(limpios)
        except ValidationError as exc:
            for campo, msg in exc.message_dict.items():
                errores.append({
                    'fila': i + 1,
                    'campo': campo,
                    'mensaje': msg if isinstance(msg, str) else msg[0],
                })
            filas_limpias.append(None)

    if errores:
        return errores, filas_limpias

    # Paso 2: correos ya existentes en la BD (una sola consulta, case-insensitive)
    correos_validos = [f['correo'] for f in filas_limpias]
    existentes = set(
        Usuario.objects
        .annotate(email_lower=Lower('email'))
        .filter(email_lower__in=correos_validos)
        .values_list('email_lower', flat=True)
    )
    for i, limpios in enumerate(filas_limpias):
        if limpios and limpios['correo'] in existentes:
            errores.append({
                'fila': i + 1,
                'campo': 'correo',
                'mensaje': 'Este correo ya está registrado.',
            })

    return errores, filas_limpias


# ──────────────────────────────────────────────────────────────────────────────
# Creación atómica en lote
# ──────────────────────────────────────────────────────────────────────────────

def crear_usuarios(escuela, rol_id: int, filas_limpias: list) -> list:
    """Crea todos los usuarios en una sola transacción atómica con bulk_create.

    Parámetros:
      escuela       — instancia de Escuela del Enlace.
      rol_id        — ROL_DOCENTE (5) o ROL_ALUMNO (6).
      filas_limpias — lista de dicts ya validados por validar_persona.

    Devuelve: lista de {'fila', 'nombre', 'correo', 'usuario', 'password'}.
    Lanza:    IntegrityError si hay carrera entre dos Enlace con el mismo correo;
              el llamador debe capturarlo y devolver 409.
    """
    tipo = TipoUsuario.objects.get(id_tipo=rol_id)
    escuela_id = escuela.id

    # Precarga usuarios que ya usan el sufijo de esta escuela (una sola consulta).
    # Evita consultar la BD por cada candidato en generar_usuario_enlace.
    sufijo = f'_{escuela_id}'
    ocupados: set[str] = set(
        Usuario.objects
        .filter(usuario__contains=sufijo)
        .values_list('usuario', flat=True)
    )

    resultado = []
    objetos = []

    for i, datos in enumerate(filas_limpias):
        password = generar_password_temporal()
        usuario = generar_usuario_enlace(
            datos['nombre'],
            escuela_id,
            lambda u, _oc=ocupados: u in _oc,
        )
        ocupados.add(usuario)  # registrar en memoria para las siguientes filas del mismo lote

        obj = Usuario(
            usuario=usuario,
            email=datos['correo'],
            nombre=datos['nombre'],
            aPaterno=datos['aPaterno'],
            aMaterno=datos['aMaterno'],
            telefono=datos['telefono'],
            tipousuario=tipo,
            escuela=escuela,
            is_active=True,
            debe_cambiar_password=True,
            # Se asigna directamente el hash ligero — no se llama a set_password()
            # para evitar el hash costoso (1 000 000 iter.) en cada fila.
            password=_hash_ligero(password),
        )
        objetos.append(obj)
        resultado.append({
            'fila': i + 1,
            'nombre': f"{datos['nombre']} {datos['aPaterno']}".strip(),
            'correo': datos['correo'],
            'usuario': usuario,
            'password': password,
        })

    with transaction.atomic():
        Usuario.objects.bulk_create(objetos)

    return resultado

