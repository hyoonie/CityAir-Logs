import time
import datetime
import requests
import pymongo
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import Dispositivo

class Command(BaseCommand):
    help = 'Worker 24/7: Consume API Redspira usando el ID en "descripcion" y guarda en Mongo.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('>>> Iniciando Redspira Worker (Clean Version) <<<'))

        # 1. CONEXIÓN A MONGODB
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
            self.stdout.write(self.style.SUCCESS(f'Conexión Mongo OK: {mongo_conf["HOST"]}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'FATAL: Fallo conexión Mongo: {e}'))
            return

        # 2. CONFIGURACIÓN API
        API_URL = "https://app.redspira.org/api/get-monitor-data"
        SLEEP_INTERVAL = 60

        # 3. BUCLE INFINITO
        while True:
            try:
                # Filtrar dispositivos con ID externo en la descripción
                dispositivos = Dispositivo.objects.exclude(
                    descripcion__isnull=True
                ).exclude(
                    descripcion__exact=''
                ).values_list('idDispositivo', 'descripcion')
                
                count = dispositivos.count()
                self.stdout.write(f"--- Ciclo de escaneo: {count} dispositivos ---")

                for internal_id, external_id_raw in dispositivos:
                    redspira_id = str(external_id_raw).strip()
                    
                    if not redspira_id:
                        continue

                    try:
                        # [cite_start]Parámetros según documentación [cite: 82, 86]
                        params = {
                            'idmonitor': redspira_id,
                            'interval': 'realtime', 
                            'aqi_method': 'aqi',
                            'lang': 'es',
                            'parameterid': '*' 
                        }
                        
                        response = requests.get(API_URL, params=params, timeout=10)
                        
                        if response.status_code == 200:
                            data_json = response.json()
                            
                            # [cite_start]Validar respuesta [cite: 92]
                            if 'data' in data_json and isinstance(data_json['data'], list) and len(data_json['data']) > 0:
                                
                                doc_to_save = {
                                    'idDispositivo': internal_id,
                                    'idRedspira': redspira_id,
                                    'timestamp': datetime.datetime.now(),
                                    'origen': 'API_REDSPIRA',
                                    'raw_data': {} 
                                }

                                for item in data_json['data']:
                                    param = item.get('parameterid')
                                    val = item.get('val')
                                    
                                    if param:
                                        doc_to_save[param] = val
                                        # Guardar metadatos adicionales en raw_data
                                        doc_to_save['raw_data'][param] = {
                                            'unit': item.get('unit'),
                                            'aqi': item.get('aqi'),
                                            'color': item.get('color', None),
                                            'desc': item.get('desc', None)
                                        }

                                collection.insert_one(doc_to_save)
                                self.stdout.write(f" > OK: {internal_id} (Redspira: {redspira_id})")
                            else:
                                self.stdout.write(self.style.WARNING(f" > VACIO: {internal_id} (ID '{redspira_id}' no retorna datos)"))
                        else:
                            self.stdout.write(self.style.ERROR(f" > HTTP {response.status_code}: {internal_id}"))

                    except Exception as e_req:
                        self.stdout.write(self.style.ERROR(f" > Error procesando {internal_id}: {e_req}"))
                    
                    time.sleep(1)

            except Exception as e_global:
                self.stdout.write(self.style.ERROR(f"CRITICAL LOOP ERROR: {e_global}"))
                time.sleep(10)

            self.stdout.write(f"Esperando {SLEEP_INTERVAL}s...")
            time.sleep(SLEEP_INTERVAL)