from django.shortcuts import render, redirect
import os
import pandas as pd
import re 
import json 
from django.http import JsonResponse 
from django.conf import settings 
from django.core.management import call_command
from django.templatetags.static import static 

# Usamos la ruta portable globalmente
CSV_PRED = settings.RUTA_DATOS_MODELO 

# ----------------------------------------------------------------------
# FUNCIÓN AUXILIAR PARA CARGAR MÉTRICAS Y RUTAS
# ----------------------------------------------------------------------

def _load_model_kpis(target='PM2_5'):
    """
    Carga las métricas (MAE, RMSE, R2) y la ruta de la imagen generadas por el modelo 
    desde el archivo TXT y define el nombre del archivo PNG estático.
    """
    
    # 1. Definir rutas de los archivos generados
    OUT_DIR = settings.OUT_DIR
    PNG_FILENAME = f"pred_vs_real_{target}.png"
    METRICS_FILENAME = f"metricas_{target}.txt"
    
    METRICS_PATH_FULL = os.path.join(OUT_DIR, METRICS_FILENAME)
    
    # 2. Valores por defecto
    data = {
        "MAE": "N/A", 
        "RMSE": "N/A", 
        "R2": "N/A", 
        "png_filename": PNG_FILENAME, 
        "error_msg": None,
        "is_data_loaded": False
    }

    try:
        # 3. Leer Métricas del archivo TXT
        if os.path.exists(METRICS_PATH_FULL):
            with open(METRICS_PATH_FULL, 'r', encoding='utf-8') as f:
                report = f.read()
                
            # Extraer valores de prueba (test) del informe de Random Forest usando expresiones regulares
            mae_match = re.search(r'== Random Forest ==.*?MAE\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            rmse_match = re.search(r'== Random Forest ==.*?RMSE\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            r2_match = re.search(r'== Random Forest ==.*?R2\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            
            if r2_match and mae_match and rmse_match:
                data['R2'] = r2_match.group(1)
                data['MAE'] = mae_match.group(1)
                data['RMSE'] = rmse_match.group(1)
                data['is_data_loaded'] = True
            else:
                data["error_msg"] = "Archivo de métricas encontrado, pero el formato no es el esperado (R2/MAE/RMSE)."
        else:
            data["error_msg"] = f"Archivos de modelo para {target} no encontrados. Ejecuta el Generador de Modelos."
        
    except Exception as e:
        data["error_msg"] = f"Error general al cargar KPIs: {e}"
        
    return data

# ----------------------------------------------------------------------
# VISTAS DE DJANGO
# ----------------------------------------------------------------------

def home(request):
    """Muestra la página de inicio/landing page (con KPIs)."""
    # Cargamos los KPIs y el estado
    ctx = _load_model_kpis(target='PM2_5') 
    return render(request, "dashboard/home.html", ctx) 
    
def analisis_view(request):
    """Carga los KPIs y la imagen de predicción para la página de Análisis."""
    ctx = _load_model_kpis(target='PM2_5') 
    ctx['target_name'] = 'PM2.5'
    
    return render(request, "dashboard/analisis_datos.html", ctx)


def quienes(request):
    return render(request, "dashboard/quienes.html")


def generador(request):
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
    
    return render(request, "dashboard/generador.html", ctx)


def resumen(request):
    # Ya que el CSV_PRED usa settings.RUTA_DATOS_MODELO, solo cargamos los datos.
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
            error_msg = f"No se encontró el archivo: {CSV_PRED}" 
    except Exception as e:
        error_msg = f"Error leyendo CSV: {e}"

    # Combinamos el contexto de la carga del CSV con los KPIs cargados del TXT
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
    return render(request, "dashboard/resumen.html", ctx)


def get_kpis_api(request):
    """
    Vista API que devuelve las métricas del modelo y el estado en formato JSON.
    Esta vista es llamada por el JavaScript en analisis_datos.html.
    """
    data = _load_model_kpis(target='PM2_5')
    
    # Prepara la URL estática que JS usará
    data['png_url'] = static(f"dashboard/{data['png_filename']}")
    
    return JsonResponse(data)