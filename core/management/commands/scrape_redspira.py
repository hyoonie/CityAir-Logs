# core/management/commands/scrape_redspira.py

import requests
from bs4 import BeautifulSoup
import re
import time # 👈 1. Importa la librería time
from django.core.management.base import BaseCommand
from django.db.models import Max
from django.utils import timezone
from core.models import Dispositivo, Lecturas # Asegúrate del nombre correcto

# --- CONFIGURACIÓN ---
DEVICE_URL = "http://192.168.4.1/?request=read"
DEVICE_DB_ID = 2
INTERVALO_SEGUNDOS = 30 # 👈 Define el intervalo aquí

# --- FUNCIONES AUXILIARES ---
def extract_float(text_value):
    # ... (tu función extract_float sin cambios) ...
    if text_value is None: return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(text_value))
    if match:
        try: return float(match.group(0))
        except ValueError: return None
    return None

# --- COMANDO DE DJANGO ---
class Command(BaseCommand):
    help = f'Obtiene datos del dispositivo Redspira cada {INTERVALO_SEGUNDOS} segundos.'

    # ✅ Mueve la lógica principal a una función separada
    def fetch_and_save_data(self):
        self.stdout.write(f"Intentando obtener datos de {DEVICE_URL}...")
        try:
            response = requests.get(DEVICE_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- Extracción (sin cambios) ---
            pm25_td = soup.find('td', string=re.compile(r'PM25:'))
            pm10_td = soup.find('td', string=re.compile(r'PM10:'))
            temp_td = soup.find('td', string=re.compile(r'Temp:'))
            hr_td = soup.find('td', string=re.compile(r'HR/RH:'))
            pa_td = soup.find('td', string=re.compile(r'PA/AP:'))
            # co2_td = soup.find(...)
            # luz_td = soup.find(...)

            pm25_raw = pm25_td.find_next_sibling('td').find('b').text.strip() if pm25_td and pm25_td.find_next_sibling('td') and pm25_td.find_next_sibling('td').find('b') else None
            pm10_raw = pm10_td.find_next_sibling('td').find('b').text.strip() if pm10_td and pm10_td.find_next_sibling('td') and pm10_td.find_next_sibling('td').find('b') else None
            temp_raw = temp_td.find_next_sibling('td').text.strip() if temp_td and temp_td.find_next_sibling('td') else None
            hr_raw = hr_td.find_next_sibling('td').text.strip() if hr_td and hr_td.find_next_sibling('td') else None
            pa_raw = pa_td.find_next_sibling('td').text.strip() if pa_td and pa_td.find_next_sibling('td') else None
            # co2_raw = ...
            # luz_raw = ...

            self.stdout.write(f"Datos crudos: PM2.5='{pm25_raw}', PM10='{pm10_raw}', Temp='{temp_raw}', HR='{hr_raw}', PA='{pa_raw}'")

            # --- Limpieza (sin cambios) ---
            pm25 = extract_float(pm25_raw)
            pm10 = extract_float(pm10_raw)
            temp = extract_float(temp_raw)
            hum = extract_float(hr_raw)
            presion = extract_float(pa_raw)
            # co2 = extract_float(co2_raw)
            # luz = extract_float(luz_raw)

            if pm25 is None and pm10 is None and temp is None and hum is None:
                 self.stderr.write(self.style.WARNING("No se pudieron extraer datos válidos esta vez."))
                 return # Continúa al siguiente ciclo

            self.stdout.write(f"Datos limpios: PM2.5={pm25}, PM10={pm10}, Temp={temp}, Hum={hum}, Presion={presion}")

            # --- Búsqueda Dispositivo (sin cambios) ---
            try:
                dispositivo = Dispositivo.objects.get(idDispositivo=DEVICE_DB_ID)
            except Dispositivo.DoesNotExist:
                self.stderr.write(self.style.ERROR(f"Dispositivo ID {DEVICE_DB_ID} no encontrado."))
                return # Detiene esta iteración si el dispositivo no existe

            # --- Calcular ID y Guardar (sin cambios) ---
            last_id_result = Lecturas.objects.aggregate(max_id=Max('idLectura'))
            last_id = last_id_result.get('max_id')
            next_id = 1 if last_id is None else last_id + 1

            nueva_lectura = Lecturas.objects.create(
                idLectura=next_id,
                dispositivo=dispositivo,
                fecha=timezone.now(),
                PM2_5=pm25,
                # PM10=pm10,
                temperatura=temp,
                humedad=hum,
                presion=presion,
                # CO2=co2,
                # luz=luz,
            )
            self.stdout.write(self.style.SUCCESS(f"Lectura guardada (ID: {nueva_lectura.idLectura})"))

        except requests.exceptions.Timeout:
             self.stderr.write(self.style.WARNING(f"Timeout al conectar con {DEVICE_URL}. Reintentando en {INTERVALO_SEGUNDOS}s..."))
        except requests.exceptions.RequestException as e:
            self.stderr.write(self.style.ERROR(f"Error de conexión: {e}"))
        except Dispositivo.DoesNotExist: # Captura por si acaso get falla
             self.stderr.write(self.style.ERROR(f"Dispositivo ID {DEVICE_DB_ID} no encontrado durante el guardado."))
        except Exception as e:
            # Captura cualquier otro error para que el bucle no se rompa
            self.stderr.write(self.style.ERROR(f"Error inesperado procesando datos: {e}"))


    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Iniciando scrapeo cada {INTERVALO_SEGUNDOS} segundos... Presiona Ctrl+C para detener."))
        
        # ✅ 2. Bucle infinito
        while True:
            try:
                # Llama a la función que hace el trabajo
                self.fetch_and_save_data()
                
                # ✅ 3. Pausa antes de la siguiente ejecución
                self.stdout.write(f"Esperando {INTERVALO_SEGUNDOS} segundos...")
                time.sleep(INTERVALO_SEGUNDOS)
                
            except KeyboardInterrupt:
                # ✅ 4. Permite detener el script con Ctrl+C
                self.stdout.write(self.style.WARNING("\nDeteniendo el script..."))
                break # Sale del bucle while