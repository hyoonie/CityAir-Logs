from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.templatetags.static import static
from django.core.management import call_command

import os
import pandas as pd
import plotly.graph_objects as go

CSV_PRED = getattr(settings, 'RUTA_DATOS_MODELO', '')

# ----------------- VISTAS TEMPLATES -----------------

def view_ct_analisis_home(request):
    """Pantalla inicial del módulo de análisis de datos"""
    ctx = _load_model_kpis(target='PM2_5')
    return render(request, 'analisis_rl_rf/analisis_home.html', ctx)

def view_ct_analisis(request):
    """Vista principal de análisis RF vs Linear"""
    ctx = _load_model_kpis(target='PM2_5')
    ctx['target_name'] = 'PM2.5'
    return render(request, 'analisis_datos_algoritmos/analisis_datos.html', ctx)

def view_ct_generador(request):
    """Vista generador de modelos"""
    if request.method == 'POST':
        try:
            target_to_run = request.POST.get('target', 'PM2_5')
            call_command('run_model', target=target_to_run)
            mensaje = f"✅ Modelos para {target_to_run} ejecutados con éxito."
            clase = 'success'
            ultima_ejecucion = 'Ahora'
        except Exception as e:
            mensaje = f"❌ Error al ejecutar el modelo: {e}"
            clase = 'danger'
            ultima_ejecucion = '—'
    else:
        mensaje = 'Presiona Generar para actualizar las predicciones.'
        clase = 'info'
        ultima_ejecucion = '—'

    ctx = {
        'mensaje': mensaje,
        'clase': clase,
        'ultima_ejecucion': ultima_ejecucion
    }
    return render(request, 'analisis_datos_algoritmos/generador.html', ctx)

def view_ct_resumen(request):
    """Resumen general de resultados"""
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
            if 'datetime' in cols:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M')

            sample = df.head(50).copy()
            table_rows = sample.to_dict(orient='records')

            plot = df.head(50)
            chart_labels = plot['datetime'].tolist() if 'datetime' in cols else list(range(len(plot)))

            col_true = next((c for c in cols if c.startswith('y_true_')), None)
            col_rf = next((c for c in cols if c.startswith('yhat_rf_')), None)

            if col_true and col_rf:
                chart_real = plot[col_true].tolist()
                chart_rf = plot[col_rf].tolist()
            else:
                error_msg = 'No se encontraron columnas de predicción en el CSV.'
        else:
            error_msg = f'No se encontró el archivo CSV: {CSV_PRED}'
    except Exception as e:
        error_msg = f'Error leyendo CSV: {e}'

    ctx = {
        'error_msg': error_msg,
        'table_rows': table_rows,
        'chart_labels': chart_labels,
        'chart_real': chart_real,
        'chart_rf': chart_rf,
        'columns': df.columns.tolist() if isinstance(df, pd.DataFrame) else [],
        'MAE': kpis['RF']['MAE'],
        'RMSE': kpis['RF']['RMSE'],
        'R2': kpis['RF']['R2'],
        'MAE_lin': kpis['Linear']['MAE'],
        'RMSE_lin': kpis['Linear']['RMSE'],
        'R2_lin': kpis['Linear']['R2'],
        'png_url_rf': static(f"dashboard/{kpis['RF']['png_filename']}"),
        'png_url_lin': static(f"dashboard/{kpis['Linear']['png_filename']}"),
    }
    return render(request, 'analisis_datos_algoritmos/resumen.html', ctx)

def view_ct_metricas(request):
    """Vista de métricas detalladas"""
    pm_data = _load_model_kpis(target='PM2_5')
    co2_data = _load_model_kpis(target='CO2')

    for d in [pm_data, co2_data]:
        d['RF']['png_url'] = static(f"dashboard/{d['RF']['png_filename']}")
        d['Linear']['png_url'] = static(f"dashboard/{d['Linear']['png_filename']}")

    ctx = {
        'pm_data': pm_data,
        'co2_data': co2_data
    }
    return render(request, 'analisis_datos_algoritmos/metricas.html', ctx)

def view_ct_analisis_mongo(request):
    """Vista de análisis MongoDB"""
    return render(request, 'analisis_datos_mongodb/analisis_mongo.html')

def view_ct_analisis_csv(request):
    """Vista de análisis CSV interactivo"""
    return render(request, 'analisis_rl_rf/analisis_csv.html')


# ----------------- API / UTIL -----------------

def api_get_kpis(request):
    pm = _load_model_kpis(target='PM2_5')
    co2 = _load_model_kpis(target='CO2')

    for data in [pm, co2]:
        for model in ['RF', 'Linear']:
            data[model]['png_url'] = static(f"dashboard/{data[model]['png_filename']}")

    return JsonResponse({'PM2_5': pm, 'CO2': co2})

def _load_model_kpis(target):
    if target == 'PM2_5':
        return {
            'RF': {'MAE': 0.840, 'RMSE': 1.058, 'R2': 0.868, 'png_filename': 'pm25_rf.png'},
            'Linear': {'MAE': 1.420, 'RMSE': 1.781, 'R2': 0.627, 'png_filename': 'pm25_lin.png'}
        }
    elif target == 'CO2':
        return {
            'RF': {'MAE': 6.236, 'RMSE': 8.184, 'R2': 0.968, 'png_filename': 'co2_rf.png'},
            'Linear': {'MAE': 24.031, 'RMSE': 30.364, 'R2': 0.566, 'png_filename': 'co2_lin.png'}
        }

# ----------------- Gráficas Interactivas -----------------
def view_ct_analisis_csv(request):
    graficas = {}

    for TARGET in ["PM2_5", "CO2"]:
        csv_path = os.path.join(
            settings.OUT_DIR,
            f"predicciones_{TARGET}.csv"
        )

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception as e:
            graficas[TARGET] = f"<p>Error leyendo CSV ({TARGET}): {e}</p>"
            continue

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[f"y_true_{TARGET}"],
            name="Valor real",
            line=dict(color="black", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[f"yhat_lin_{TARGET}"],
            name="Regresión Lineal",
            line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[f"yhat_rf_{TARGET}"],
            name="Random Forest"
        ))

        fig.update_layout(
            title=f"Comparación Real vs Predicciones ({TARGET})",
            xaxis_title="Fecha",
            yaxis_title=TARGET,
            hovermode="x unified"
        )

        graficas[TARGET] = fig.to_html(full_html=False)

    return render(
        request,
        "analisis_rl_rf/analisis_csv.html",
        {
            "grafica_pm25": graficas.get("PM2_5", ""),
            "grafica_co2": graficas.get("CO2", ""),
        }
    )
