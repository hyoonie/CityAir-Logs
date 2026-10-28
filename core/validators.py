import re

from django.core.exceptions import ValidationError

# CCT de México: entidad (01-32) + 3 letras (sostenimiento y tipo de servicio)
# + 4 dígitos de folio + letra verificadora. Ej. 30EPR0001X
CCT_MEXICO_REGEX = re.compile(r'^(0[1-9]|[12]\d|3[0-2])[A-Z]{3}\d{4}[A-Z]$')

# 10 dígitos, con o sin separadores (espacio o guion) en los cortes 3-3-4
TELEFONO_REGEX = re.compile(r'^\d{3}[\s-]?\d{3}[\s-]?\d{4}$')

MENSAJE_CCT = ('La clave CCT no tiene un formato válido (ej. 30EPR0001X: '
               '2 dígitos de entidad, 3 letras, 4 dígitos y 1 letra).')
MENSAJE_TELEFONO = 'El teléfono debe tener 10 dígitos con el formato 228-165-0525.'


def validar_cct_mexico(valor):
    if not CCT_MEXICO_REGEX.match(valor or ''):
        raise ValidationError(MENSAJE_CCT, code='cct_invalida')


def normalizar_telefono(valor):
    """Acepta '2281650525', '228 165 0525' o '228-165-0525' y devuelve '228-165-0525'."""
    valor = (valor or '').strip()
    if not TELEFONO_REGEX.match(valor):
        raise ValidationError(MENSAJE_TELEFONO, code='telefono_invalido')
    digitos = re.sub(r'\D', '', valor)
    return f'{digitos[:3]}-{digitos[3:6]}-{digitos[6:]}'
