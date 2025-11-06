from django.shortcuts import render, redirect
import os
import pandas as pd
# import re  <-- Ya no se necesita aquí
import json 
from django.http import JsonResponse 
from django.conf import settings 
from django.core.management import call_command
from django.templatetags.static import static 

# --- LÍNEAS AÑADIDAS QUE FALTABAN ---
# 1. Importa la lógica de 'utils.py' (Corrige el NameError)
from .utils import _load_model_kpis
# 2. Define la variable CSV_PRED que usa 'view_resumen'
CSV_PRED = getattr(settings, 'RUTA_DATOS_MODELO', '') 
# --- FIN DE LÍNEAS AÑADIDAS ---


# Aqui se definen las vistas de los html dentro de core/templates...

#Home
def view_home(request):
    # Asegúrate que esta es la vista de inicio que quieres.
    # Esta vista está apuntando a 'home.html' (la plantilla base genérica)
    # Quizás querías que apunte a 'view_home_analisis_datos'
     return render(request, 'home.html')

# core/template/sesiones
def view_ct_login(request):
     return render(request, 'sesiones/login.html')

# core/template/about us
def view_aboutus(request):
    return render(request, 'about_us/about_us.html')

# core/template/datos
def view_ct_analisis(request):
    return render(request, 'analisis/analisis.html')

# core/template/dispositivos
def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

# core/template/reportes
def view_ct_reportes(request):
     return render(request, 'reportes/reportes.html')

# core/template/alertas
def view_ct_alertas(request):
     return render(request, 'alertas/alertas.html')

#################### Jared
def view_home_analisis_datos(request):
     """Muestra la página de inicio (analisis_datos_home.html)"""
     # Llama a la lógica importada
     ctx = _load_model_kpis(target='PM2_5') 
     return render(request, 'analisis_datos_algoritmos/analisis_datos_home.html', ctx)

def view_analisis_datos(request):
     """Carga los KPIs y la imagen de predicción para la página de Análisis."""
     # Llama a la lógica importada
     ctx = _load_model_kpis(target='PM2_5') 
     ctx['target_name'] = 'PM2.5'
     return render(request, 'analisis_datos_algoritmos/analisis_datos.html', ctx)

def view_generador(request):
     """Ejecuta el comando de gestión para generar nuevos datos y métricas."""
     if request.method == 'POST':
        try:
            target_to_run = request.POST.get('target', 'PM2_5')
            call_command('run_model', target=target_to_run)
            mensaje = f"✅ Modelos para {target_to_run} ejecutados con éxito. ¡Refresca el Resumen!"
            clase = 'success'
        except Exception as e:
            mensaje = f"❌ ERROR al ejecutar el modelo: {e}"
            clase = 'danger'
        
        ctx = {'mensaje': mensaje, 'clase': clase, 'ultima_ejecucion': 'Ahora'}
     
     else:
        ctx = {'mensaje': 'Presiona Generar para actualizar las predicciones.', 'clase': 'info'}
     
     return render(request, 'analisis_datos_algoritmos/generador.html', ctx)

def view_resumen(request):
     # Llama a la lógica importada
     kpis = _load_model_kpis(target='PM2_5')

     df = None
     table_rows = []
     chart_labels = []
     chart_real = []
     chart_rf = []
     error_msg = None

     try:
         if os.path.exists(CSV_PRED):
             df = pd.read_csv(CSV_PRED)
             cols = df.columns.tolist()
             if "datetime" in cols:
                 df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
             
             sample = df.head(50).copy()
             table_rows = sample.to_dict(orient="records")

             plot = df.head(50)
             chart_labels = plot["datetime"].tolist() if "datetime" in cols else list(range(len(plot)))
             col_true = next((c for c in cols if c.startswith("y_true_")), None)
             col_rf   = next((c for c in cols if c.startswith("yhat_rf_")), None)
             
             if col_true and col_rf:
                 chart_real = plot[col_true].tolist()
                 chart_rf   = plot[col_rf].tolist()
             else:
                 error_msg = "No se encontraron columnas de predicción esperadas en el CSV."
         else:
             # Proporciona un mensaje de error más claro que incluye la ruta esperada
             error_msg = f"No se encontró el archivo: {CSV_PRED}. (Ruta esperada: {CSV_PRED})" 
     except Exception as e:
         error_msg = f"Error leyendo CSV: {e}"

     ctx = {
         "error_msg": error_msg,
         "table_rows": table_rows,
         "chart_labels": chart_labels,
         "chart_real": chart_real,
         "chart_rf": chart_rf,
         "columns": df.columns.tolist() if isinstance(df, pd.DataFrame) else [],
         "MAE": kpis['MAE'], 
         "RMSE": kpis['RMSE'],
         "R2": kpis['R2'],
     }
     # Apunta a la plantilla en la carpeta correcta
     return render(request, "analisis_datos_algoritmos/resumen.html", ctx)


#
# === FUNCIÓN DE API ===
#
def api_get_kpis(request):
    """
    Vista API que devuelve las métricas del modelo y el estado en formato JSON.
    """
    # Llama a la lógica importada
    data = _load_model_kpis(target='PM2_5')
    
    # Prepara la URL estática que JS usará
    # (Asegúrate que 'dashboard/' es la subcarpeta correcta en 'static/')
    data['png_url'] = static(f"dashboard/{data['png_filename']}")
    
    return JsonResponse(data)