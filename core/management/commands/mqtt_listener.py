import paho.mqtt.client as mqtt
import json
import os
import datetime
import pytz
import pymongo
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from core.models import Dispositivo

# ==========================================
# 1. CONFIGURACIÓN MONGO DB
# ==========================================
try:
    mongo_conf = settings.MONGO_SETTINGS
    client_mongo = pymongo.MongoClient(
        host=mongo_conf['HOST'],
        port=int(mongo_conf['PORT']),
        username=mongo_conf['USERNAME'],
        password=mongo_conf['PASSWORD'],
        authSource=mongo_conf.get('AUTH_SOURCE', 'admin'),
        authMechanism=mongo_conf.get('AUTH_MECHANISM', 'SCRAM-SHA-1')
    )
    db = client_mongo[mongo_conf['DB_NAME']]
    collection_name = os.getenv("MONGO_COLLECTION", "lecturas")
    mongo_collection = db[collection_name]
    print(f"✓ Conexión a MongoDB establecida. Colección: {collection_name}")
except Exception as e:
    print(f"⚠ Error crítico al conectar con MongoDB: {e}")
    mongo_collection = None

# ==========================================
# 2. CONFIGURACIÓN TTN
# ==========================================
TTN_HOST = os.getenv("TTN_HOST", "au1.cloud.thethings.network")
TTN_PORT = int(os.getenv("TTN_PORT", 8883))
TTN_USERNAME = os.getenv("TTN_USERNAME")
TTN_PASSWORD = os.getenv("TTN_PASSWORD")
TTN_TOPIC = f"v3/{TTN_USERNAME}@ttn/devices/+/up"

TIMEZONE_MAP = {
    'mexico': 'America/Mexico_City',
    'colombia': 'America/Bogota',
}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado exitosamente al broker de TTN!")
        client.subscribe(TTN_TOPIC)
        print(f"Suscrito al topic: {TTN_TOPIC}")
    else:
        print(f"Fallo al conectar, código de error: {rc}")

def get_timezone_for_device(dispositivo):
    if dispositivo.pais:
        pais_lower = dispositivo.pais.lower()
        timezone_str = TIMEZONE_MAP.get(pais_lower) 
        if timezone_str:
            try:
                return pytz.timezone(timezone_str)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"ADVERTENCIA: Zona horaria '{timezone_str}' no reconocida.")
    return pytz.utc

def on_message(client, userdata, msg):
    # Validar conexión a Mongo antes de procesar
    if mongo_collection is None:
        print("Error: No hay conexión a MongoDB. Omitiendo mensaje.")
        return

    try:
        # Decodificar payload
        data = json.loads(msg.payload.decode('utf-8'))
        device_eui = data.get("end_device_ids", {}).get("dev_eui")
        payload_data = data.get("uplink_message", {}).get("decoded_payload")
        
        if not device_eui or not payload_data:
            print("Mensaje incompleto (falta EUI o payload). Ignorando.")
            return

        # =================================================================
        # CONSULTA SQL: Validar dispositivo y descripción TTN
        # =================================================================
        try:
            # Buscamos el dispositivo por su dev_eui único
            # Filtramos TAMBIÉN por descripcion='TTN' como solicitaste
            dispositivo = Dispositivo.objects.get(dev_eui=device_eui, descripcion='TTN')
            
            # Recuperamos el ID real de la base de datos (Primary Key)
            real_db_id = dispositivo.idDispositivo
            print(f"✓ Dispositivo encontrado en BD: {real_db_id} (EUI: {device_eui})")

        except Dispositivo.DoesNotExist:
            print(f"⚠ Ignorando EUI {device_eui}: No existe en BD o su descripción no es 'TTN'.")
            return

        # =================================================================
        # PROCESAMIENTO
        # =================================================================
        
        # 1. Cálculo de fecha con zona horaria del dispositivo
        device_timezone = get_timezone_for_device(dispositivo)
        now_utc = datetime.datetime.now(pytz.utc)
        now_local_naive = now_utc.astimezone(device_timezone).replace(tzinfo=None)
        fecha_mongo = make_aware(now_local_naive, timezone=pytz.utc)

        # 2. Construcción del Documento
        # Usamos 'real_db_id' (el idDispositivo de MySQL) en lugar del ID de TTN
        documento_lectura = {
            "fecha": fecha_mongo,
            "idDispositivo": real_db_id,   # ID recuperado de MySQL
            "temperatura": payload_data.get("TEM"),
            "humedad": payload_data.get("HUM"),
            "presion": payload_data.get("pressure"),
            "metrica": "nom025",
            "PM2_5": payload_data.get("PM2_5"),
            "PM10": payload_data.get("PM10"),
            "CO2": payload_data.get("CO2"),
            "luz": payload_data.get("illumination")
        }

        # 3. Guardado en Mongo
        result = mongo_collection.insert_one(documento_lectura)
        print(f"Lectura guardada. OID: {result.inserted_id}")

    except json.JSONDecodeError:
        print("Error al decodificar JSON del mensaje MQTT.")
    except Exception as e:
        print(f"Error inesperado al procesar mensaje: {e}")

class Command(BaseCommand):
    help = 'Inicia el cliente MQTT -> MongoDB (Solo dispositivos TTN)'

    def handle(self, *args, **options):
        if not TTN_USERNAME or not TTN_PASSWORD:
            self.stdout.write(self.style.ERROR('Faltan credenciales TTN en .env'))
            return

        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.username_pw_set(TTN_USERNAME, TTN_PASSWORD) 
        client.tls_set()
        
        self.stdout.write(self.style.SUCCESS('Iniciando cliente MQTT...'))
        
        try:
            client.connect(TTN_HOST, TTN_PORT, 60)
            client.loop_forever()
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('Cliente detenido.'))
            client.disconnect()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error de conexión: {e}'))