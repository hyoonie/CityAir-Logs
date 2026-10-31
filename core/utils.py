import secrets
import pymongo
from CityAirLogs import settings

def get_db_collection():
    db_config = settings.MONGO_SETTINGS

    client = pymongo.MongoClient(
        host=db_config["HOST"],
        port=db_config["PORT"],
        username=db_config["USERNAME"],
        password=db_config["PASSWORD"],
        authSource=db_config["AUTH_SOURCE"],
        authMechanism=db_config.get("AUTH_MECHANISM", "SCRAM-SHA-1"),
    )

    db = client[db_config["DB_NAME"]]

    return client, db["lecturas"]


def get_distinct_devices():
    """Obtiene lista única de dispositivos para el checkbox"""
    client, collection = get_db_collection()
    devices = collection.distinct("idDispositivo")
    client.close()
    return sorted([d for d in devices if d])


def get_mongo_data(query_filter):
    """Ejecuta la consulta directamente en Mongo"""
    client, collection = get_db_collection()
    data = list(collection.find(query_filter, {"_id": 0}))
    client.close()
    return data

# ── Contraseñas temporales ───────────────────────────────────────────────────
# Se usa el módulo `secrets` y NO `random`: `random` es un generador predecible
# (con unas cuantas salidas se puede deducir la siguiente) y está pensado para
# simulaciones. `secrets` usa la fuente de aleatoriedad criptográfica del sistema
# operativo, que es la apropiada para contraseñas y tokens.
#
# Se quitaron los caracteres ambiguos (0/O, 1/l/I) porque el administrador tiene
# que dictar o copiar la contraseña a otra persona, y esos se confunden fácil.
_MAYUS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
_MINUS = 'abcdefghijkmnpqrstuvwxyz'
_DIGITOS = '23456789'
_SIMBOLOS = '!@#$%*?'


#Genera una contraseña temporal segura (12 caracteres por defecto).
# Cómo funciona:
# 1. Toma un carácter de CADA grupo (mayúscula, minúscula, dígito, símbolo), para
# garantizar que siempre cumpla variedad.
# 2. Rellena el resto con caracteres al azar de todos los grupos.
# 3. Mezcla el resultado. Sin este paso, los primeros 4 caracteres siempre
# tendrían el mismo orden (mayúscula, minúscula, dígito, símbolo) y la
# contraseña sería más fácil de adivinar.
# 
# Elegi implementarlo en utils.py porque es una función pura, no consulta la base de datos ni
# depende de ningún usuario, a diferencia de generar_usuario_enlace, que sí
# revisa colisiones y por eso está en el Manager.
def generar_password_temporal(longitud=12):
    """Genera una contraseña temporal segura con al menos un carácter de cada tipo."""
    todos = _MAYUS + _MINUS + _DIGITOS + _SIMBOLOS
    chars = [secrets.choice(g) for g in (_MAYUS, _MINUS, _DIGITOS, _SIMBOLOS)]
    chars += [secrets.choice(todos) for _ in range(longitud - 4)]
    secrets.SystemRandom().shuffle(chars)
    return ''.join(chars)