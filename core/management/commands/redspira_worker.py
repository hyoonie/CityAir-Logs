import time
import datetime
import requests
import json
import pymongo
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import Dispositivo

class Command(BaseCommand):
    help = 'Worker V5.1: Scrape de JSON dispositivos Redspira a MongoDB-CityAirLogs'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('>>> INICIANDO WORKER V5.1 (FINAL) <<<'))

        mongo_conf = settings.MONGO_SETTINGS
        try:
            client = pymongo.MongoClient(
                host=mongo_conf['HOST'],
                port=mongo_conf['PORT'],
                username=mongo_conf['USERNAME'],
                password=mongo_conf['PASSWORD'],
                authSource=mongo_conf['AUTH_SOURCE'],
                authMechanism=mongo_conf['AUTH_MECHANISM']
            )
            db = client[mongo_conf['DB_NAME']]
            collection = db['lecturas']
            self.stdout.write(self.style.SUCCESS(f'[DB] Conectado a {mongo_conf["HOST"]}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'[DB] FATAL: {e}'))
            return

        API_URL = "https://app.redspira.org/api/get-monitor-data"
        SLEEP_INTERVAL = 60 
        
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        KEY_MAPPING = {
            'temp': 'temperatura',
            'hr':   'humedad',
            'pa':   'presion',
            'pm25': 'PM2_5',
            'pm10': 'PM10',         
            'co2':  'CO2',
            'o3':   'O3'
        }

        while True:
            try:
                dispositivos = Dispositivo.objects.exclude(descripcion__isnull=True).exclude(descripcion__exact='').values_list('idDispositivo', 'descripcion')
                
                print("\n" + "="*60)
                self.stdout.write(f"--- NUEVO BARRIDO: {dispositivos.count()} dispositivos ---")

                now = datetime.datetime.now()
                fecha_str = now.strftime("%Y-%m-%d")
                hora_calculada = now.hour - 1
                if hora_calculada < 0: hora_calculada = 23
                
                print(f"--- Params: Date='{fecha_str}', Time='{hora_calculada}' ---")

                for internal_id, external_id_raw in dispositivos:
                    redspira_id = str(external_id_raw).strip()
                    if not redspira_id: continue

                    print(f"\n> Procesando: [{internal_id}] ({redspira_id})")

                    try:
                        params = {
                            'idmonitor': redspira_id,
                            'date': fecha_str,
                            'time': hora_calculada,
                            'timezone': '-6',
                            'parameterid': '*',
                            'interval': 'realtime',
                            'aqi_method': 'nom025',
                            'lang': 'es'
                        }
                        
                        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)

                        if response.status_code == 200:
                            try:
                                data_json = response.json()
                                
                                if 'data' in data_json and isinstance(data_json['data'], list) and len(data_json['data']) > 0:
                                    doc_to_save = {
                                        'fecha': datetime.datetime.now(),
                                        'idDispositivo': internal_id,
                                        'idRedspira': redspira_id,
                                        'metrica': params['aqi_method'] 
                                    }
                                    
                                    log_info = []

                                    for item in data_json['data']:
                                        raw_param = item.get('parameterid')
                                        val = item.get('val')
                                        
                                        if raw_param: 
                                            std_key = KEY_MAPPING.get(raw_param, raw_param.upper())
                                            
                                            doc_to_save[std_key] = val
                                            
                                            if std_key in ['temperatura', 'PM2_5', 'humedad']:
                                                log_info.append(f"{std_key}:{val}")

                                    collection.insert_one(doc_to_save)
                                    self.stdout.write(self.style.SUCCESS(f"  [EXITO] Guardado: {', '.join(log_info)}"))
                                
                                else:
                                    self.stdout.write(self.style.WARNING(f"  [ALERTA] Data vacía."))

                            except json.JSONDecodeError:
                                self.stdout.write(self.style.ERROR(f"  [ERROR] No es JSON válido."))
                        else:
                            self.stdout.write(self.style.ERROR(f"  [ERROR HTTP] {response.status_code}"))

                    except Exception as e_req:
                        self.stdout.write(self.style.ERROR(f"  [EXCEPCION] {e_req}"))
                    
                    time.sleep(1)

            except Exception as e_global:
                self.stdout.write(self.style.ERROR(f"CRITICAL ERROR: {e_global}"))
                time.sleep(10)

            print(f"\nEsperando {SLEEP_INTERVAL}s...")
            time.sleep(SLEEP_INTERVAL)