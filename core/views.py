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
                user_type = user.tipousuario.nombre_tipo
                print(f"El tipo de usuario es: '{user_type}'")
                print(f"¿Es admin?: {user_type == 'Administrador de sistema'}")
                print(f"¿Es investigador?: {user_type == 'Investigador/Analista'}")
                print(f"¿Es user?: {user_type == 'Usuario general'}")
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
        # --- FIX IS HERE ---
        user = serializer.save() # 1. Capture the created user instance
        return redirect('cuenta', user_id=user.id) # 2. Pass the ID to the URL
        # -------------------

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
            kmeans_plot_data = [] # [FIX] Inicializar vacíos
            plotly_layout = {}    # [FIX] Inicializar vacíos
            if len(df_hourly_mean) >= 3:
                df_ai = df_hourly_mean.bfill().fillna(0)
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df_ai)
                
                n_clusters = 3 if len(df_ai) > 10 else 1
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_features)
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                
                df_hourly_mean['Cluster'] = clusters
                
                # --- NOMBRADO INTELIGENTE (TU LÓGICA ANTERIOR) ---
                if 'PM2_5' in df_hourly_mean.columns:
                    rank_score = df_hourly_mean.groupby('Cluster')['PM2_5'].mean()
                else:
                    rank_score = df_hourly_mean.groupby('Cluster').mean().mean(axis=1)
                
                sorted_indices = rank_score.sort_values().index.tolist()
                
                labels_map = {}
                definitions = ["Contaminación Baja", "Contaminación Media", "Contaminación Alta"]
                
                for i, c_id in enumerate(sorted_indices):
                    if i < len(definitions): labels_map[c_id] = definitions[i]
                    else: labels_map[c_id] = f"Perfil {i+1}"

                # --- [NUEVO] PREPARACIÓN DE DATOS PARA PLOTLY 3D ---
                # Seleccionamos 3 variables para los ejes X, Y, Z
                axes_cols = []
                ejes_nombres = []
                
                # Prioridad: Temp, Humedad, PM2.5
                if 'temperatura' in df_hourly_mean.columns: axes_cols.append('temperatura'); ejes_nombres.append('Temp (°C)')
                if 'humedad' in df_hourly_mean.columns: axes_cols.append('humedad'); ejes_nombres.append('Hum (%)')
                if 'PM2_5' in df_hourly_mean.columns: axes_cols.append('PM2_5'); ejes_nombres.append('PM2.5')
                elif 'CO2' in df_hourly_mean.columns: axes_cols.append('CO2'); ejes_nombres.append('CO2')

                kmeans_plot_data = []
                
                # Solo graficamos si tenemos al menos 3 variables
                if len(axes_cols) >= 3:
                    for c_id in sorted(df_hourly_mean['Cluster'].unique()):
                        cluster_data = df_hourly_mean[df_hourly_mean['Cluster'] == c_id]
                        label_nombre = labels_map.get(c_id, f'Cluster {c_id}')
                        
                        # Asignar color según el nombre
                        color_hex = '#198754' # Verde (Baja)
                        if 'Media' in label_nombre: color_hex = '#ffc107' # Amarillo
                        elif 'Alta' in label_nombre: color_hex = '#dc3545' # Rojo
                        
                        trace = {
                            'x': cluster_data[axes_cols[0]].tolist(),
                            'y': cluster_data[axes_cols[1]].tolist(),
                            'z': cluster_data[axes_cols[2]].tolist(),
                            'mode': 'markers',
                            'type': 'scatter3d',
                            'name': label_nombre,
                            'marker': {'size': 4, 'color': color_hex, 'opacity': 0.8}
                        }
                        kmeans_plot_data.append(trace)
                
                plotly_layout = {
                    'xaxis_title': ejes_nombres[0] if len(ejes_nombres) > 0 else 'X',
                    'yaxis_title': ejes_nombres[1] if len(ejes_nombres) > 1 else 'Y',
                    'zaxis_title': ejes_nombres[2] if len(ejes_nombres) > 2 else 'Z',
                }

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
                'narrativa': narrativa_textos,
                'kmeans_plot_data': json.dumps(kmeans_plot_data),
                'plotly_layout_titles': json.dumps(plotly_layout)
            })

        except Exception as e:
            print("ERROR EN VISTA:")
            traceback.print_exc()
            context['error_message'] = f"Error: {str(e)}"

    return render(request, 'datos/datos.html', context)

#narrativa de resultados
def generar_narrativa_inteligente(quality_report, df, analysis_summary):
    """
    Genera un informe narrativo detallado y didáctico sobre el análisis de datos.
    Desmenuza la información de calidad, normativa, IA y tendencias visuales.
    """
    narrativa = []
    
    # --- 1. DETALLE DE CALIDAD DE DATOS ---
    integrity = quality_report.get('integrity_pct', 0)
    rows_raw = quality_report.get('total_rows_raw', 0)
    rows_valid = rows_raw - quality_report.get('rows_dropped', 0)
    
    texto_calidad = f"<strong>1. Confiabilidad del Estudio:</strong><br>"
    texto_calidad += f"Para este análisis, el sistema recolectó un total de <strong>{rows_raw} lecturas brutas</strong> de los dispositivos seleccionados. "
    texto_calidad += f"Tras aplicar los filtros de limpieza, se conservaron <strong>{rows_valid} registros válidos</strong> para el cálculo matemático. "
    
    if integrity >= 95:
        texto_calidad += f"Esto representa una integridad del <span class='text-success fw-bold'>{integrity}% (Excelente)</span>. "
        texto_calidad += "Significa que la comunicación con los sensores fue constante y no hubo pérdidas significativas de información, por lo que las conclusiones presentadas a continuación tienen un nivel de confianza muy alto."
    elif integrity >= 80:
        texto_calidad += f"La integridad de los datos es del <span class='text-warning fw-bold'>{integrity}% (Aceptable)</span>. "
        texto_calidad += "Aunque existe una base sólida para el análisis, se detectaron algunos huecos en la información (posibles desconexiones momentáneas), los cuales fueron rellenados estadísticamente para no afectar las gráficas."
    else:
        texto_calidad += f"La integridad de los datos es <span class='text-danger fw-bold'>Baja ({integrity}%)</span>. "
        texto_calidad += "Se perdió una cantidad considerable de información. Se recomienda verificar la conexión a internet de los dispositivos o su suministro de energía, ya que esto podría ocultar eventos importantes."
    
    narrativa.append(texto_calidad)

    # --- 2. EXPLICACIÓN NORMATIVA (PM2.5) ---
    if 'PM2_5' in df.columns:
        pm25_mean = df['PM2_5'].mean()
        pm25_max = df['PM2_5'].max()
        
        texto_norma = f"<strong>2. Evaluación de Salud (Partículas PM2.5):</strong><br>"
        texto_norma += "Las partículas finas (PM2.5) son el contaminante más crítico para la salud respiratoria. "
        texto_norma += f"Durante el periodo analizado, la concentración promedio fue de <strong>{round(pm25_mean, 2)} µg/m³</strong>. "
        
        # Comparativa NOM-025
        if pm25_mean <= 15:
            texto_norma += "Al comparar este valor con la norma oficial mexicana <strong>NOM-025-SSA1-2021</strong>, el aire se clasifica como <span class='badge bg-success'>BUENO</span> (0-15 µg/m³). "
            texto_norma += "Esto indica un ambiente limpio, ideal para realizar cualquier tipo de actividad al aire libre sin riesgos para la salud."
        elif pm25_mean <= 33:
            texto_norma += "Este nivel entra en la categoría de <span class='badge bg-warning text-dark'>ACEPTABLE</span> (>15-33 µg/m³). "
            texto_norma += "Aunque cumple con el límite máximo permisible, existe un riesgo moderado solamente para personas extremadamente sensibles (asmáticos o con enfermedades cardíacas), quienes deberían reducir esfuerzos prolongados."
        else:
            texto_norma += "<strong>¡Atención!</strong> Este valor <span class='badge bg-danger'>SUPERA LA NORMA</span> (>33 µg/m³), entrando en categoría de MALA calidad. "
            texto_norma += "Respirar aire con esta concentración de forma continua puede incrementar el riesgo de enfermedades respiratorias. Se recomienda a la población sensible evitar actividades físicas vigorosas en el exterior."
            
        texto_norma += f"<br>Cabe destacar que, en su punto más crítico, la contaminación alcanzó un pico de <strong>{round(pm25_max, 2)} µg/m³</strong>."
        narrativa.append(texto_norma)

    # --- 3. INTERPRETACIÓN DE INTELIGENCIA ARTIFICIAL ---
    if analysis_summary:
        texto_ia = f"<strong>3. Hallazgos del análisis de datos:</strong><br>"
        
        # Clusters
        c0 = analysis_summary.get('cluster_0_count', 0)
        c1 = analysis_summary.get('cluster_1_count', 0)
        c2 = analysis_summary.get('cluster_2_count', 0)
        total_horas = analysis_summary.get('total_hours', 1)
        
        counts = [c0, c1, c2]
        labels = [
            analysis_summary.get('cluster_0_label', 'Perfil 1'),
            analysis_summary.get('cluster_1_label', 'Perfil 2'),
            analysis_summary.get('cluster_2_label', 'Perfil 3')
        ]
        
        # Ordenamos para explicar del más presente al menos presente
        sorted_profiles = sorted(zip(counts, labels), reverse=True)
        ganador_count, ganador_label = sorted_profiles[0]
        ganador_pct = round((ganador_count / total_horas) * 100, 1)

        texto_ia += f"El algoritmo de agrupamiento (K-Means) analizó el comportamiento conjunto de todas las variables (Temp, Humedad, Contaminantes) y detectó que el escenario predominante fue de <strong>'{ganador_label}'</strong>. "
        texto_ia += f"Este perfil de comportamiento se mantuvo constante durante el <strong>{ganador_pct}%</strong> del tiempo monitoreado."
        
        # Anomalías
        anomalias = analysis_summary.get('anomalies_count', 0)
        if anomalias > 0:
            texto_ia += f"<br><br>Por otro lado, el detector de anomalías (DBSCAN) identificó <strong>{anomalias} momentos atípicos</strong>. "
            texto_ia += "Estos son eventos puntuales donde las lecturas de los sensores se salieron drásticamente del patrón habitual (picos repentinos de ruido o contaminación) y merecen una revisión manual detallada en las gráficas."
        else:
            texto_ia += "<br><br>El sistema se comportó de manera muy estable. No se detectaron anomalías estadísticas ni eventos 'raros' fuera de los parámetros normales."
        
        narrativa.append(texto_ia)

    # --- 4. DESGLOSE DETALLADO DE GRÁFICAS (TENDENCIAS) ---
    variables_a_explicar = [col for col in df.columns if col in ['temperatura', 'humedad', 'CO2', 'luz']]
    
    if variables_a_explicar:
        detalles_graficas = "<strong>4. Interpretación Visual de las Variables:</strong><br>"
        detalles_graficas += "A continuación se desglosa el comportamiento observado en las gráficas de línea, tomar en cuenta que las gráficas están promediadas por hora:<br><ul class='mb-0'>"
        
        for col in variables_a_explicar:
            # Cálculos estadísticos
            v_min = df[col].min()
            v_max = df[col].max()
            v_mean = df[col].mean()
            t_max = df[col].idxmax() # Fecha del pico
            t_min = df[col].idxmin() # Fecha del mínimo
            
            # Análisis de volatilidad
            std_dev = df[col].std()
            coef_var = (std_dev / v_mean) * 100 if v_mean != 0 else 0
            
            estabilidad = ""
            if coef_var < 5: 
                estabilidad = "extremadamente estable, prácticamente una línea plana"
            elif coef_var < 15: 
                estabilidad = "estable, con variaciones naturales suaves"
            elif coef_var < 30: 
                estabilidad = "variable, con fluctuaciones claras a lo largo del día"
            else: 
                estabilidad = "altamente volátil, con cambios bruscos y repentinos"

            # Nombre amigable y unidades
            nombre_var = col.capitalize()
            unidad = ""
            if col == 'temperatura': unidad = "°C"
            elif col == 'humedad': unidad = "%"
            elif col == 'CO2': nombre_var = 'CO2'; unidad = " ppm"
            elif col == 'luz': unidad = " lux"

            # Redacción del punto
            frase = f"<li><strong>{nombre_var}:</strong> Mostró un comportamiento <em>{estabilidad}</em>. "
            frase += f"El promedio se situó en {round(v_mean, 1)}{unidad}. "
            frase += f"Se observa que el valor descendió hasta un mínimo de <strong>{round(v_min, 1)}{unidad}</strong> (registrado el {t_min.strftime('%d/%m %H:%M')}) "
            frase += f"y alcanzó su punto más alto de <strong>{round(v_max, 1)}{unidad}</strong> el día {t_max.strftime('%d/%m a las %H:%M')}.</li>"
            
            detalles_graficas += frase
        
        detalles_graficas += "</ul>"
        narrativa.append(detalles_graficas)

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