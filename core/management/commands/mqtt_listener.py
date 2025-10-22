# core/management/commands/mqtt_listener.py

import paho.mqtt.client as mqtt
import json, os, datetime, pytz
from dotenv import load_dotenv
from django.core.management.base import BaseCommand
from django.db.models import Max
from django.utils.timezone import make_aware
from core.models import Dispositivo, Lecturas

# Creds TTN
load_dotenv()
TTN_HOST = os.getenv("TTN_HOST", "au1.cloud.thethings.network")
TTN_PORT = int(os.getenv("TTN_PORT", 8883))
TTN_USERNAME = os.getenv("TTN_USERNAME")
TTN_PASSWORD = os.getenv("TTN_PASSWORD")
TTN_TOPIC = f"v3/{TTN_USERNAME}@ttn/devices/+/up"

# Timezones para fecha de lectura
TIMEZONE_MAP = {
    'mexico': 'America/Mexico_City',
    'colombia': 'America/Bogota',
    # Añade más aquí:
}

# Logica Cliente MQTT: Conexion y Lectura TTN
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
        # Busca en el diccionario
        timezone_str = TIMEZONE_MAP.get(pais_lower) 
        if timezone_str:
            print(f"Zona horaria encontrada en mapa para '{pais_lower}': {timezone_str}")
            try:
                return pytz.timezone(timezone_str)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"ADVERTENCIA: Zona horaria '{timezone_str}' no reconocida.")
        else:
            print("DEBUG: No se encontró país en el dispositivo.")

    # Si no se encuentra en el mapa, usa UTC
    print(f"ADVERTENCIA: No se encontró zona horaria mapeada para el país '{dispositivo.pais}', usando UTC.")
    return pytz.utc

# Esta función es la más importante. Se ejecuta cada vez que llega un mensaje.
def on_message(client, userdata, msg):
    
    print(f"Mensaje recibido en el topic: {msg.topic}")
    
    # Decodifica el payload del dispostivo
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        device_eui = data.get("end_device_ids", {}).get("dev_eui")
        payload_data = data.get("uplink_message", {}).get("decoded_payload")
        
        if not device_eui or not payload_data:
            print("Mensaje inválido. Ignorando.")
            return

        print(f"Datos del EUI '{device_eui}': {payload_data}")

        # Busca o crea el dispositivo
        dispositivo, created = Dispositivo.objects.get_or_create(dev_eui=device_eui)
        if created:
            print(f"Dispositivo con EUI '{device_eui}' creado.")

        # CÁLCULO DE FECHA UTC (Por dispositivo)
        device_timezone = get_timezone_for_device(dispositivo)
        now_utc = datetime.datetime.now(pytz.utc)
        now_local_naive = now_utc.astimezone(device_timezone).replace(tzinfo=None)
        print(f"Fecha/Hora Ingenua calculada ({device_timezone}): {now_local_naive.strftime('%Y-%m-%d %H:%M:%S')}")
        fecha_para_guardar = make_aware(now_local_naive, timezone=pytz.utc)
        print(f"Fecha que se enviará a Django (UTC forzado): {fecha_para_guardar}")

        # Calcular el siguiente idLectura
        last_id_result = Lecturas.objects.aggregate(max_id=Max('idLectura'))
        last_id = last_id_result.get('max_id')
        next_id = 1 if last_id is None else last_id + 1
        print(f"Asignando idLectura: {next_id}")

        Lecturas.objects.create(
            idLectura=next_id,
            dispositivo=dispositivo,
            fecha=fecha_para_guardar,
            temperatura=payload_data.get("TEM"), 
            humedad=payload_data.get("HUM"),
            PM2_5=payload_data.get("PM2_5"), 
            CO2=payload_data.get("CO2"),
            luz=payload_data.get("illumination"),
            presion=payload_data.get("pressure"),
        )
        print(f"Nueva Lectura guardada (ID: {next_id})")

    except json.JSONDecodeError:
        print("Error al decodificar JSON.")
    except Dispositivo.DoesNotExist: # Más específico que Exception general
        print(f"Error: Dispositivo con EUI {device_eui} no encontrado y no se pudo crear.")
    except KeyError as e: # Error común si falta una clave en payload_data
        print(f"Error: Falta la clave '{e}' en los datos decodificados.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

# COMANDO DE DJANGO
class Command(BaseCommand):
    help = 'Inicia el cliente MQTT...'

    def handle(self, *args, **options):
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
            self.stdout.write(self.style.WARNING('Cliente MQTT detenido.'))
            client.disconnect()
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'No se pudo conectar al broker MQTT: {e}'))