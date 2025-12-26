import traceback
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
            print(f"¿Tiene ID asignado?: {user.id}")

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
            {"error": "El rol 'usuario general' no está configurado en el sistema."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # Crear una copia mutable del request.data y añadir el rol
    data = request.data.copy()  # Copiar los datos del request (mutable)
    data['tipousuario_id'] = tipo_user.id_tipo  # Añadir el rol al diccionario
    
    # Crear el usuario con los datos proporcionados en el request
    serializer = UsuarioSerializer(data=data)
    
    if serializer.is_valid():
        serializer.save()
        return redirect('cuenta')  # Redirige a la vista del login
    return Response(serializer.errors, status=400)

# core/template/about us
def view_aboutus(request):
        return render(request, 'about_us/about_us.html')

# core/template/datos
def view_ct_datos(request):
    # 1. OBTENER TODOS LOS DISPOSITIVOS (Sin filtrar)
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
            # Seguridad: Si envían más de 3, solo tomamos los primeros 3 para el análisis
            selected_devs = request.POST.getlist('devices')[:3] 
            selected_vars = request.POST.getlist('variables')
            start_date_str = request.POST.get('start_date')
            end_date_str = request.POST.get('end_date')

            # Validación: Fechas obligatorias
            if not start_date_str or not end_date_str:
                context['error_message'] = "Por favor selecciona un Rango de Fechas válido."
                context['selected_devs'] = selected_devs # Mantener selección
                return render(request, 'datos/datos.html', context)

            context.update({
                'selected_devs': selected_devs, 'selected_vars': selected_vars,
                'start_date': start_date_str, 'end_date': end_date_str
            })

            # --- QUERY MONGO ---
            mongo_query = {}
            if selected_devs:
                mongo_query['idDispositivo'] = {'$in': selected_devs}

            try:
                start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)
                mongo_query['fecha'] = {'$gte': start_dt, '$lte': end_dt}
            except ValueError: pass

            print(f"--- NUEVA CONSULTA: {mongo_query}")
            data_list = get_mongo_data(mongo_query)
            
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
            
            # --- REPORTE DE CALIDAD ---
            total_rows_raw = len(df)
            quality_stats = {}
            
            for col in available_cols_in_db:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                final_nans = df[col].isna().sum()
                quality_stats[col] = int(final_nans)

            rows_before_drop = len(df)
            df = df.dropna(subset=available_cols_in_db, how='all')
            rows_dropped = rows_before_drop - len(df)
            if df.empty: raise ValueError("Sin datos numéricos válidos.")

            df_hourly_check = df[available_cols_in_db].resample('h').mean() 
            imputed_counts = {}

            if not df_hourly_check.empty:
                for col in available_cols_in_db:
                    imputed_counts[col] = int(df_hourly_check[col].isna().sum())
            
            quality_report = { 
                'total_rows_raw': total_rows_raw,
                'rows_dropped': rows_dropped,
                'integrity_pct': round((1 - (rows_dropped / total_rows_raw)) * 100, 1) if total_rows_raw > 0 else 0,
                'invalid_per_col': quality_stats,
                'imputed_per_col': imputed_counts
            }

            # --- PREPARACIÓN GRÁFICAS ---
            df_pivot = df.pivot_table(index='fecha', columns='idDispositivo', values=available_cols_in_db, aggfunc='mean')
            df_hourly_pivot = df_pivot.resample('h').mean()
            
            df_hourly_mean = df[available_cols_in_db].resample('h').mean().dropna(how='all')
            if df_hourly_mean.empty and len(df) > 0: 
                df_hourly_mean = df[available_cols_in_db].copy()

            charts_data = {}
            time_labels = df_hourly_pivot.index.strftime('%Y-%m-%d %H:%M').tolist()
            vars_to_graph = selected_vars if selected_vars else available_cols_in_db
            found_devices = df_hourly_pivot.columns.get_level_values(1).unique().tolist()

            for col in available_cols_in_db:
                if col in vars_to_graph:
                    datasets = []
                    all_values = []
                    for dev in found_devices:
                        if (col, dev) in df_hourly_pivot.columns:
                            series = df_hourly_pivot[(col, dev)]
                            raw_values = series.replace({np.nan: None}).tolist()
                            valid_data = [v for v in raw_values if v is not None]
                            if valid_data:
                                datasets.append({'label': dev, 'data': raw_values})
                                all_values.extend(valid_data)
                    
                    if datasets:
                        g_min = min(all_values) if all_values else 0
                        g_max = max(all_values) if all_values else 0
                        g_mean = sum(all_values)/len(all_values) if all_values else 0
                        charts_data[col] = {
                            'labels': time_labels, 'datasets': datasets,
                            'min': round(g_min, 2), 'max': round(g_max, 2), 'mean': round(g_mean, 2)
                        }

            # --- ANÁLISIS DATOS ---
            analysis_summary = None
            if len(df_hourly_mean) >= 3:
                df_ai = df_hourly_mean.bfill().fillna(0)
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df_ai)
                
                n_clusters = 3 if len(df_ai) > 10 else 1
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_features)
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                
                df_hourly_mean['Cluster'] = clusters
                
                # --- NOMBRADO INTELIGENTE DE CLUSTERS ---
                # 1. Calculamos el promedio de contaminación de cada cluster
                if 'PM2_5' in df_hourly_mean.columns:
                    rank_score = df_hourly_mean.groupby('Cluster')['PM2_5'].mean()
                else:
                    rank_score = df_hourly_mean.groupby('Cluster').mean().mean(axis=1)
                
                # 2. Ordenamos de menor a mayor contaminación
                sorted_indices = rank_score.sort_values().index.tolist()
                
                # 3. Asignamos nombres legibles
                labels_map = {}
                definitions = ["Contaminación Baja", "Contaminación Media", "Contaminación Alta"]
                
                for i, c_id in enumerate(sorted_indices):
                    if i < len(definitions):
                        labels_map[c_id] = definitions[i]
                    else:
                        labels_map[c_id] = f"Perfil {i+1}"

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
                             p = p if p > 0 else 1.0
                             t = t if t > 0 else 1.0
                             h = h if h > 0 else 1.0
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
                    # Enviamos las etiquetas al HTML
                    'cluster_0_label': labels_map.get(0, 'Cluster 0'),
                    'cluster_1_label': labels_map.get(1, 'Cluster 1'),
                    'cluster_2_label': labels_map.get(2, 'Cluster 2'),
                    'anomalies_count': n_anomalies,
                    'dea_data': dea_report
                }
            narrativa_textos = generar_narrativa_inteligente(quality_report, df, analysis_summary)

            context.update({
                'show_results': True,
                'charts_data': json.dumps(charts_data),
                'available_cols': list(charts_data.keys()),
                'analysis': analysis_summary,
                'quality_report': quality_report,
                'quality_json': json.dumps(quality_report),
                'narrativa': narrativa_textos
            })

        except Exception as e:
            print("ERROR EN VISTA:")
            traceback.print_exc()
            context['error_message'] = f"Error: {str(e)}"

    return render(request, 'datos/datos.html', context)

#narrativa de resultados
def generar_narrativa_inteligente(quality_report, df, analysis_summary):
    """
    Toma los datos estadísticos y genera una lista de párrafos explicativos 
    en lenguaje natural para el usuario.
    """
    narrativa = []
    
    # --- 1. BLOQUE DE CALIDAD DE DATOS ---
    integrity = quality_report.get('integrity_pct', 0)
    rows_raw = quality_report.get('total_rows_raw', 0)
    rows_valid = rows_raw - quality_report.get('rows_dropped', 0)
    
    texto_calidad = f"<strong>Análisis de Integridad:</strong> El sistema procesó un total de {rows_raw} registros, de los cuales {rows_valid} fueron válidos. "
    
    if integrity >= 95:
        texto_calidad += f"La calidad de los datos es <span class='text-success fw-bold'>Excelente ({integrity}%)</span>. Los resultados presentados tienen un nivel de confianza muy alto."
    elif integrity >= 80:
        texto_calidad += f"La calidad de los datos es <span class='text-warning fw-bold'>Aceptable ({integrity}%)</span>. Se realizaron imputaciones menores para mantener la continuidad del análisis."
    else:
        texto_calidad += f"La calidad de los datos es <span class='text-danger fw-bold'>Baja ({integrity}%)</span>. Se recomienda revisar el estado de los sensores, ya que la pérdida de datos podría afectar la precisión de las conclusiones."
    narrativa.append(texto_calidad)

    # --- 2. BLOQUE DE NORMATIVA (PM2.5) ---
    if 'PM2_5' in df.columns:
        pm25_mean = df['PM2_5'].mean()
        pm25_max = df['PM2_5'].max()
        
        texto_norma = f"<strong>Cumplimiento Normativo (PM2.5):</strong> El promedio global durante el periodo fue de <strong>{round(pm25_mean, 2)} µg/m³</strong>. "
        
        # Lógica basada en NOM-025-SSA1-2021
        if pm25_mean <= 15:
            texto_norma += "Este valor se encuentra dentro del rango de <span class='badge bg-success'>Calidad Buena</span> (0-15 µg/m³), lo que indica un ambiente saludable y libre de riesgos significativos."
        elif pm25_mean <= 33:
            texto_norma += "Este valor clasifica como <span class='badge bg-warning text-dark'>Aceptable</span> (>15-33 µg/m³). Existe un riesgo moderado para grupos extremadamente sensibles, pero es aceptable para la población general."
        else:
            texto_norma += "⚠️ Este valor <strong>EXCEDE</strong> el límite máximo permisible de 33 µg/m³, categorizándose como <span class='badge bg-danger'>Mala</span>. Se recomienda evitar actividades vigorosas al aire libre."
            
        texto_norma += f" Se registró un pico máximo de contaminación de {round(pm25_max, 2)} µg/m³."
        narrativa.append(texto_norma)

    # --- 3. BLOQUE DE INTELIGENCIA ARTIFICIAL ---
    if analysis_summary:
        # A. Clusters (Perfiles)
        c0 = analysis_summary.get('cluster_0_count', 0)
        c1 = analysis_summary.get('cluster_1_count', 0)
        c2 = analysis_summary.get('cluster_2_count', 0)
        total_horas = analysis_summary.get('total_hours', 1)
        
        # Encontrar cuál ganó
        counts = [c0, c1, c2]
        labels = [
            analysis_summary.get('cluster_0_label', 'Perfil 1'),
            analysis_summary.get('cluster_1_label', 'Perfil 2'),
            analysis_summary.get('cluster_2_label', 'Perfil 3')
        ]
        max_idx = counts.index(max(counts))
        ganador_label = labels[max_idx]
        ganador_pct = round((counts[max_idx] / total_horas) * 100, 1)

        texto_ia = f"<strong>Patrones Detectados (IA):</strong> El algoritmo K-Means identificó que el comportamiento predominante del aire corresponde a <strong>'{ganador_label}'</strong>, presente durante el {ganador_pct}% del tiempo analizado. "
        
        # B. Anomalías
        anomalias = analysis_summary.get('anomalies_count', 0)
        if anomalias > 0:
            texto_ia += f"Sin embargo, el sistema de detección de anomalías (DBSCAN) encontró <strong>{anomalias} eventos atípicos</strong>. Estos son momentos donde los sensores arrojaron valores que no coinciden con los patrones normales y requieren revisión manual."
        else:
            texto_ia += "El comportamiento de los sensores fue estable y consistente, sin detectar anomalías o lecturas extrañas."
        
        narrativa.append(texto_ia)

        # C. Punto Óptimo (DEA)
        dea = analysis_summary.get('dea_data')
        if dea:
            texto_dea = f"<strong>Recomendación de Eficiencia:</strong> Basado en el Análisis Envolvente de Datos (DEA), el momento con la mejor relación calidad/confort se registró el <strong>{dea.get('best_date')}</strong>, con una concentración de PM2.5 de apenas {dea.get('pm25_val')} µg/m³."
            narrativa.append(texto_dea)

    return narrativa    

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