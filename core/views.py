from django.shortcuts import render, redirect
import os
import pandas as pd
from django.http import JsonResponse 
from django.conf import settings 
from django.core.management import call_command
from django.templatetags.static import static 

# --- Variables de configuración ---
CSV_PRED = getattr(settings, 'RUTA_DATOS_MODELO', '')

# ----------------- VISTAS -----------------

def view_home(request):
    return render(request, 'home.html')

def view_ct_login(request):
    return render(request, 'sesiones/login.html')

def view_aboutus(request):
    return render(request, 'about_us/about_us.html')

def view_ct_analisis(request):
    return render(request, 'analisis/analisis.html')

def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

def view_ct_reportes(request):
    return render(request, 'reportes/reportes.html')

def view_ct_alertas(request):
    return render(request, 'alertas/alertas.html')


# ----------------- ANÁLISIS DE DATOS -----------------

def view_home_analisis_datos(request):
    ctx = _load_model_kpis(target='PM2_5')
    return render(request, 'analisis_datos_algoritmos/analisis_datos_home.html', ctx)

def view_analisis_datos(request):
    ctx = _load_model_kpis(target='PM2_5')
    ctx['target_name'] = 'PM2.5'
    return render(request, 'analisis_datos_algoritmos/analisis_datos.html', ctx)

def view_generador(request):
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

        # KPIs Random Forest
        "MAE": kpis['RF']['MAE'],
        "RMSE": kpis['RF']['RMSE'],
        "R2": kpis['RF']['R2'],

        # KPIs Regresión Lineal
        "MAE_lin": kpis['Linear']['MAE'],
        "RMSE_lin": kpis['Linear']['RMSE'],
        "R2_lin": kpis['Linear']['R2'],

        # URLs de gráficos
        "png_url_rf": static(f"dashboard/{kpis['RF']['png_filename']}"),
        "png_url_lin": static(f"dashboard/{kpis['Linear']['png_filename']}"),
    }
    return render(request, "analisis_datos_algoritmos/resumen.html", ctx)


# ----------------- API para JS -----------------

def api_get_kpis(request):
    pm = _load_model_kpis(target='PM2_5')
    co2 = _load_model_kpis(target='CO2')

    # Solo agregamos png_url plano para que JS funcione
    for data in [pm, co2]:
        for model in ['RF', 'Linear']:
            data[model]['png_url'] = static(f"dashboard/{data[model]['png_filename']}")

    return JsonResponse({
        "PM2_5": pm,
        "CO2": co2
    })


# ----------------- MÉTRICAS DETALLADAS -----------------

def view_metricas(request):
    pm_data = _load_model_kpis(target='PM2_5')
    co2_data = _load_model_kpis(target='CO2')

    # Añadir URLs para los gráficos en la plantilla de métricas
    for d in [pm_data, co2_data]:
        d["RF"]["png_url"] = static(f"dashboard/{d['RF']['png_filename']}")
        d["Linear"]["png_url"] = static(f"dashboard/{d['Linear']['png_filename']}")

    context = {
        "pm_data": pm_data,
        "co2_data": co2_data
    }

    return render(request, "analisis_datos_algoritmos/metricas.html", context)


# ----------------- UTIL -----------------

def _load_model_kpis(target):
    """
    Devuelve métricas de Random Forest y Regresión Lineal para un target.
    """
    if target == "PM2_5":
        return {
            "RF": {
                "MAE": 0.840,
                "RMSE": 1.058,
                "R2": 0.868,
                "png_filename": "pm25_rf.png"
            },
            "Linear": {
                "MAE": 1.420,
                "RMSE": 1.781,
                "R2": 0.627,
                "png_filename": "pm25_lin.png"
            }
        }
    elif target == "CO2":
        return {
            "RF": {
                "MAE": 6.236,
                "RMSE": 8.184,
                "R2": 0.968,
                "png_filename": "co2_rf.png"
            },
            "Linear": {
                "MAE": 24.031,
                "RMSE": 30.364,
                "R2": 0.566,
                "png_filename": "co2_lin.png"
            }
        }
