import pymongo
from datetime import datetime, timedelta
from django.shortcuts import render, redirect
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import logout
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from rest_framework import status
from django.contrib import messages
from django.http import HttpResponseForbidden
from django.urls import reverse
from django.db.models import Count, Q
from django.utils import timezone
import pandas as pd
import numpy as np
from django.shortcuts import render

from CityAirLogs import settings
from .forms import UploadFileForm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import json

from .models import TipoUsuario, Usuario, Enfermedad, Dispositivo
from .serializers import *

# Aqui se definen las vistas de los html dentro de core/templates...

#Home
def view_home(request):
    return render(request, 'home.html')

# core/template/sesiones
def view_ct_login(request):
    # Si la petición es POST, procesamos el formulario
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            print(f"--- Intento de login para: {user.usuario} ---")
            print(f"¿Tiene tipousuario asignado?: {user.tipousuario}")

            if user.tipousuario:
                user_type = user.tipousuario.nombre_tipo()
                print(f"El tipo de usuario es: '{user_type}'")
                print(f"¿Es admin?: {user_type == 'Administrador de sistema'}")
                print(f"¿Es investigador?: {user_type == 'Investigador/Analista'}")
                print(f"¿Es user?: {user_type == 'Usuario general'}")
                if user_type == 'Administrador de sistema':
                    return redirect('homeAdmin')
                elif user_type == 'Usuario general':
                    return redirect('cuenta', user_id=user.id)
            
            # Si el usuario no tiene un tipo o es uno no reconocido, lo mandamos al login
            return redirect('login')
    
    # Si la petición es GET, o si el formulario del POST fue inválido
    else:
        form = AuthenticationForm()
        
    return render(request, 'sesiones/login.html', {'form': form})

# core/template/sesiones/cuenta
def view_profile(request, user_id):
    usuario = get_object_or_404(Usuario, id=user_id)
    return render(request, 'sesiones/cuenta/cuenta.html', {
        'usuario': usuario,
    })

# core/template/sesiones/registro
def view_cts_register(request):
     return render(request, 'sesiones/registros/registro.html')
@api_view(['POST'])
def registro_usuarios(request):
    # Asignar el rol "cliente" directamente en la vista
    try:
        tipo_user = TipoUsuario.objects.get(nombre_tipo="Usuario general")
    except TipoUsuario.DoesNotExist:
        return Response(
            {"error": "El rol 'user' no está configurado en el sistema."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # Crear una copia mutable del request.data y añadir el rol
    data = request.data.copy()  # Copiar los datos del request (mutable)
    data['tipousuario_id'] = tipo_user.id_tipo  # Añadir el rol al diccionario
    
    # Crear el usuario con los datos proporcionados en el request
    serializer = UsuarioSerializer(data=data)
    
    if serializer.is_valid():
        serializer.save()
        return redirect('login')  # Redirige a la vista del login
    return Response(serializer.errors, status=400)

# core/template/about us
def view_aboutus(request):
        return render(request, 'about_us/about_us.html')

# core/template/datos
def view_ct_datos(request):
    try:
        device_list = get_distinct_devices()
    except:
        device_list = []

    context = {
        'show_results': False, 'error_message': None, 'device_list': device_list,
        'selected_devs': [], 'selected_vars': [], 'start_date': '', 'end_date': ''
    }

    if request.method == 'POST':
        try:
            # --- CAPTURA DE INPUTS ---
            selected_devs = request.POST.getlist('devices')
            selected_vars = request.POST.getlist('variables')
            start_date_str = request.POST.get('start_date')
            end_date_str = request.POST.get('end_date')

            context.update({
                'selected_devs': selected_devs, 'selected_vars': selected_vars,
                'start_date': start_date_str, 'end_date': end_date_str
            })

            # --- QUERY MONGO ---
            mongo_query = {}
            if selected_devs:
                mongo_query['idDispositivo'] = {'$in': selected_devs}

            if start_date_str and end_date_str:
                try:
                    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d') + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
                    mongo_query['fecha'] = {'$gte': start_dt, '$lte': end_dt}
                except ValueError: pass

            print(f"--- NUEVA CONSULTA: {mongo_query}")
            data_list = get_mongo_data(mongo_query)
            print(f"Registros encontrados: {len(data_list)}")

            if not data_list:
                raise ValueError("No se encontraron registros con esos filtros.")

            df = pd.DataFrame(data_list)

            # --- NORMALIZACIÓN ---
            df.columns = df.columns.str.lower().str.strip()
            rename_map = {
                'fecha': 'fecha', 'date': 'fecha', 'timestamp': 'fecha',
                'iddispositivo': 'idDispositivo', 'deviceid': 'idDispositivo',
                'temperatura': 'temperatura', 'temperature': 'temperatura', 'temp': 'temperatura',
                'humedad': 'humedad', 'humidity': 'humedad', 'hum': 'humedad',
                'co2': 'CO2', 'eco2': 'CO2',
                'pm2_5': 'PM2_5', 'pm2.5': 'PM2_5', 'pm25': 'PM2_5',
                'presion': 'presion', 'luz': 'luz', 'light': 'luz'
            }
            new_cols = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=new_cols)

            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.set_index('fecha').sort_index()
            else: raise ValueError("No se encontró columna de fecha.")

            possible_cols = ['PM2_5', 'CO2', 'temperatura', 'humedad', 'luz']
            available_cols_in_db = [c for c in possible_cols if c in df.columns]
            
            for col in available_cols_in_db:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Limpieza básica
            df = df.dropna(subset=available_cols_in_db, how='all')
            if df.empty: raise ValueError("Sin datos numéricos válidos.")

            # =================================================================
            # ESTRATEGIA DUAL: GRÁFICAS (DETALLE) vs IA (AGREGADO)
            # =================================================================

            # 1. PREPARACIÓN PARA GRÁFICAS (Separar por Dispositivo)
            # Pivoteamos para tener columnas compuestas (Variable, Dispositivo)
            df_pivot = df.pivot_table(index='fecha', columns='idDispositivo', values=available_cols_in_db, aggfunc='mean')
            
            # Resampleamos la tabla pivoteada
            df_hourly_pivot = df_pivot.resample('h').mean()
            
            # 2. PREPARACIÓN PARA IA (Promedio General - NO TOCAR LÓGICA)
            # Este es el DF "aplanado" que usábamos antes
            df_hourly_mean = df[available_cols_in_db].resample('h').mean().dropna(how='all')
            if df_hourly_mean.empty:
                if len(df) > 0: df_hourly_mean = df[available_cols_in_db].copy() # Fallback

            # --- GENERACIÓN JSON GRÁFICAS ---
            charts_data = {}
            # Usamos el índice del pivote que incluye todos los tiempos
            time_labels = df_hourly_pivot.index.strftime('%Y-%m-%d %H:%M').tolist()
            vars_to_graph = selected_vars if selected_vars else available_cols_in_db
            
            # Detectamos qué dispositivos tienen datos en el pivote
            found_devices = df_hourly_pivot.columns.get_level_values(1).unique().tolist()

            for col in available_cols_in_db:
                if col in vars_to_graph:
                    datasets = []
                    all_values_for_stats = [] # Para calcular min/max global de la tarjeta

                    for dev in found_devices:
                        if (col, dev) in df_hourly_pivot.columns:
                            # Extraemos la serie de este dispositivo específico
                            series = df_hourly_pivot[(col, dev)]
                            raw_values = series.replace({np.nan: None}).tolist()
                            
                            # Si tiene al menos un dato, lo agregamos a la gráfica
                            valid_data = [v for v in raw_values if v is not None]
                            if valid_data:
                                datasets.append({
                                    'label': dev, # Nombre del dispositivo para la leyenda
                                    'data': raw_values
                                })
                                all_values_for_stats.extend(valid_data)
                    
                    if datasets:
                        # Estadísticas Globales (de todos los dispositivos combinados) para el footer
                        g_min = min(all_values_for_stats) if all_values_for_stats else 0
                        g_max = max(all_values_for_stats) if all_values_for_stats else 0
                        g_mean = sum(all_values_for_stats)/len(all_values_for_stats) if all_values_for_stats else 0

                        charts_data[col] = {
                            'labels': time_labels,
                            'datasets': datasets, # LISTA DE DATASETS
                            'min': round(g_min, 2),
                            'max': round(g_max, 2),
                            'mean': round(g_mean, 2)
                        }

            # --- ANÁLISIS IA (INTACTO - USA df_hourly_mean) ---
            analysis_summary = None
            if len(df_hourly_mean) >= 3:
                # Fillna con bfill para IA
                df_ai = df_hourly_mean.bfill().fillna(0)
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df_ai)
                
                # K-Means
                n_clusters = 3 if len(df_ai) > 10 else 1
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_features)
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                
                df_hourly_mean['Cluster'] = clusters
                
                # DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                anomalies = dbscan.fit_predict(scaled_features)
                n_anomalies = list(anomalies).count(-1)
                
                # DEA
                dea_report = None
                if {'PM2_5', 'temperatura', 'humedad'}.issubset(df_hourly_mean.columns):
                     def calc_dea(row):
                         try:
                             p = row.get('PM2_5', 1.0)
                             t = row.get('temperatura', 1.0)
                             h = row.get('humedad', 1.0)
                             p = p if pd.notna(p) and p > 0 else 1.0
                             t = t if pd.notna(t) and t > 0 else 1.0
                             h = h if pd.notna(h) and h > 0 else 1.0
                             return (1/p) / (h * t)
                         except: return 0
                     
                     df_hourly_mean['DEA_Score'] = df_hourly_mean.apply(calc_dea, axis=1)
                     best_hour = df_hourly_mean['DEA_Score'].idxmax()
                     
                     if pd.notna(best_hour):
                        dea_report = {
                            'best_date': str(best_hour),
                            'pm25_val': round(df_hourly_mean.loc[best_hour, 'PM2_5'], 2) if 'PM2_5' in df_hourly_mean else 0,
                            'temp_val': round(df_hourly_mean.loc[best_hour, 'temperatura'], 2) if 'temperatura' in df_hourly_mean else 0,
                            'hum_val': round(df_hourly_mean.loc[best_hour, 'humedad'], 2) if 'humedad' in df_hourly_mean else 0
                        }

                analysis_summary = {
                    'total_hours': len(df_hourly_mean),
                    'cluster_0_count': int(cluster_counts.get(0, 0)),
                    'cluster_1_count': int(cluster_counts.get(1, 0)),
                    'cluster_2_count': int(cluster_counts.get(2, 0)),
                    'anomalies_count': n_anomalies,
                    'dea_data': dea_report
                }

            context.update({
                'show_results': True,
                'charts_data': json.dumps(charts_data),
                'available_cols': list(charts_data.keys()),
                'analysis': analysis_summary
            })

        except Exception as e:
            print("ERROR EN VISTA:")
            import traceback
            traceback.print_exc()
            context['error_message'] = f"Error: {str(e)}"

    return render(request, 'datos/datos.html', context)

# core/template/dispositivos
def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

# core/template/reportes
def view_ct_reportes(request):
     return render(request, 'reportes/reportes.html')

# backend datos
def get_db_collection():
    db_config = settings.MONGO_SETTINGS
    
    client = pymongo.MongoClient(
        host=db_config['HOST'],
        port=db_config['PORT'],
        username=db_config['USERNAME'],
        password=db_config['PASSWORD'],
        authSource=db_config['AUTH_SOURCE'],
        authMechanism=db_config.get('AUTH_MECHANISM', 'SCRAM-SHA-1')
    )
    
    db = client[db_config['DB_NAME']]
    
    return client, db['lecturas']

def get_distinct_devices():
    """Obtiene lista única de dispositivos para el checkbox"""
    client, collection = get_db_collection()
    devices = collection.distinct('idDispositivo')
    client.close()
    return sorted([d for d in devices if d]) 

def get_mongo_data(query_filter):
    """Ejecuta la consulta directamente en Mongo"""
    client, collection = get_db_collection()
    data = list(collection.find(query_filter, {'_id': 0})) 
    client.close()
    return data