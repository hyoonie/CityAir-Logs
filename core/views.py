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
from .forms import UploadFileForm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import json

from .models import TipoUsuario, Usuario, Enfermedad, Dispositivo, Lecturas
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
    context = {
        'form': UploadFileForm(),
        'show_results': False,
        'error_message': None
    }

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            try:
                # 1. Carga y Limpieza Básica
                df = pd.read_csv(csv_file, delimiter=';')
                
                # Corrección de Timestamp
                if 'Fecha' in df.columns and 'Hora' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['Fecha'].astype(str) + ' ' + df['Hora'].astype(str))
                    df = df.set_index('timestamp').drop(columns=['Fecha', 'Hora'])
                else:
                    raise ValueError("El CSV debe tener columnas 'Fecha' y 'Hora'.")
                
                # Variables numéricas
                possible_cols = ['PM2_5', 'PM10', 'CO2', 'temperatura', 'humedad', 'luz']
                available_cols = [c for c in possible_cols if c in df.columns]
                
                if not available_cols:
                     raise ValueError("No se encontraron columnas válidas.")

                # Resampleo por Hora
                df_hourly = df[available_cols].resample('h').mean().dropna()
                
                # 2. Datos para Gráficas de LÍNEA (Tiempo vs Valor)
                charts_data = {}
                time_labels = df_hourly.index.strftime('%Y-%m-%d %H:%M').tolist()
                
                for col in available_cols:
                    charts_data[col] = {
                        'labels': time_labels,
                        'values': df_hourly[col].tolist(),
                        'min': round(df_hourly[col].min(), 2),
                        'max': round(df_hourly[col].max(), 2),
                        'mean': round(df_hourly[col].mean(), 2)
                    }

                # 3. Análisis Avanzado (K-Means, DBSCAN, DEA)
                if len(df_hourly) > 10:
                    # K-Means
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(df_hourly)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_features)
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    # Interpretación de Centroides
                    df_hourly['Cluster'] = clusters
                    centroids = df_hourly.groupby('Cluster').mean()
                    
                    # DBSCAN
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    anomalies = dbscan.fit_predict(scaled_features)
                    n_anomalies = list(anomalies).count(-1)
                    
                    # DEA
                    dea_report = None
                    if 'PM2_5' in df_hourly.columns and 'temperatura' in df_hourly.columns and 'humedad' in df_hourly.columns:
                         def calc_dea(row):
                             if row['PM2_5'] <= 0: return 0
                             return (1/row['PM2_5']) / (row['humedad'] * row['temperatura'])
                         
                         df_hourly['DEA_Score'] = df_hourly.apply(calc_dea, axis=1)
                         best_hour = df_hourly['DEA_Score'].idxmax()
                         dea_report = {
                             'best_date': str(best_hour),
                             'pm25_val': round(df_hourly.loc[best_hour, 'PM2_5'], 2),
                             'temp_val': round(df_hourly.loc[best_hour, 'temperatura'], 2),
                             'hum_val': round(df_hourly.loc[best_hour, 'humedad'], 2)
                         }

                    # Reporte Texto
                    analysis_summary = {
                        'total_hours': len(df_hourly),
                        'cluster_0_desc': f"Promedios: PM2.5={centroids.iloc[0].get('PM2_5',0):.1f}",
                        'cluster_0_count': int(cluster_counts.get(0, 0)),
                        'cluster_1_desc': f"Promedios: PM2.5={centroids.iloc[1].get('PM2_5',0):.1f}",
                        'cluster_1_count': int(cluster_counts.get(1, 0)),
                        'cluster_2_desc': f"Promedios: PM2.5={centroids.iloc[2].get('PM2_5',0):.1f}",
                        'cluster_2_count': int(cluster_counts.get(2, 0)),
                        'anomalies_count': n_anomalies,
                        'dea_data': dea_report
                    }
                else:
                    analysis_summary = None

                context.update({
                    'show_results': True,
                    'charts_data': json.dumps(charts_data),
                    'available_cols': available_cols,
                    'analysis': analysis_summary
                })

            except Exception as e:
                context['error_message'] = f"Error: {str(e)}"

    return render(request, 'datos/datos.html', context)

# core/template/dispositivos
def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

# core/template/reportes
def view_ct_reportes(request):
     return render(request, 'reportes/reportes.html')