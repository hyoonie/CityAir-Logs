import re
import unicodedata

from django.core.exceptions import ValidationError
from django.utils import timezone
from django.utils.crypto import get_random_string

# ──────────────────────────────────────────────
# Constantes de validación (ampliables)
# ──────────────────────────────────────────────

# Dominios de correo aceptados. Los de DOMINIOS_CON_SUBDOMINIOS también aceptan
# subdominios (p. ej. itx.edu.mx).
DOMINIOS_PERMITIDOS = ('gmail.com', 'edu.mx')
DOMINIOS_CON_SUBDOMINIOS = ('edu.mx',)

# CCT México: entidad (01-32) + 3 letras (sostenimiento y tipo de servicio)
# + 4 dígitos de folio + letra verificadora. Ej. 30EPR0001X
CCT_MEXICO_REGEX = re.compile(r'^(0[1-9]|[12]\d|3[0-2])[A-Z]{3}\d{4}[A-Z]$')

TELEFONO_REGEX = re.compile(r'^\d{3}[\s-]?\d{3}[\s-]?\d{4}$')

PDF_TAMANO_MAXIMO = 5 * 1024 * 1024  # 5 MB

# Sin caracteres ambiguos (0/O, 1/l/I) para que la contraseña se pueda dictar
CARACTERES_PASSWORD = 'abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'


def validar_cct_mexico(valor):
    if not CCT_MEXICO_REGEX.match(valor or ''):
        raise ValidationError(
            'La clave CCT no tiene un formato válido (ej. 30EPR0001X: '
            '2 dígitos de entidad, 3 letras, 4 dígitos y 1 letra).'
        )


def normalizar_telefono(valor):
    """Acepta '228 165 0525' o '228-165-0525' y devuelve '228-165-0525'."""
    valor = (valor or '').strip()
    if not TELEFONO_REGEX.match(valor):
        raise ValidationError('El teléfono debe tener 10 dígitos con el formato ###-###-####.')
    digitos = re.sub(r'\D', '', valor)
    return f'{digitos[:3]}-{digitos[3:6]}-{digitos[6:]}'


def validar_correo_dominio(correo):
    dominio = (correo or '').rsplit('@', 1)[-1].lower()
    if dominio in DOMINIOS_PERMITIDOS:
        return
    if any(dominio.endswith('.' + d) for d in DOMINIOS_CON_SUBDOMINIOS):
        return
    permitidos = ', '.join(
        ['@' + d for d in DOMINIOS_PERMITIDOS] + ['@*.' + d for d in DOMINIOS_CON_SUBDOMINIOS]
    )
    raise ValidationError(f'El dominio del correo no está permitido. Dominios aceptados: {permitidos}.')


def validar_fecha_no_futura(fecha):
    if fecha and fecha > timezone.localdate():
        raise ValidationError('La fecha de firma no puede ser posterior a la fecha actual.')


def validar_pdf(archivo):
    if not archivo.name.lower().endswith('.pdf'):
        raise ValidationError('El documento debe ser un archivo PDF.')
    if archivo.size > PDF_TAMANO_MAXIMO:
        raise ValidationError(f'El PDF no debe superar {PDF_TAMANO_MAXIMO // (1024 * 1024)} MB.')
    archivo.seek(0)
    cabecera = archivo.read(5)
    archivo.seek(0)
    if cabecera != b'%PDF-':
        raise ValidationError('El archivo no es un PDF válido.')


def generar_password_temporal(longitud=12):
    return get_random_string(longitud, CARACTERES_PASSWORD)


def _sin_acentos(texto):
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')


def generar_usuario_enlace(nombre, escuela_id, existe):
    """Genera '<primer_nombre>_<id_escuela>' (máx. 45 caracteres).
    `existe` es una función que indica si un nombre de usuario ya está ocupado."""
    primer_nombre = (nombre or '').split()[0] if (nombre or '').split() else ''
    base = re.sub(r'[^a-z0-9]', '', _sin_acentos(primer_nombre).lower()) or 'enlace'
    sufijo = f'_{escuela_id}'
    candidato = base[:45 - len(sufijo)] + sufijo
    n = 2
    while existe(candidato):
        extra = f'{sufijo}_{n}'
        candidato = base[:45 - len(extra)] + extra
        n += 1
    return candidato


def dividir_apellidos(apellidos):
    """Primera palabra -> apellido paterno; el resto -> apellido materno."""
    partes = (apellidos or '').split()
    if not partes:
        return '', ''
    return partes[0], ' '.join(partes[1:])
