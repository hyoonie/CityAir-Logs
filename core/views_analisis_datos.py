from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.templatetags.static import static
from django.core.management import call_command
from types import SimpleNamespace

import os
import pandas as pd
import plotly.graph_objects as go

CSV_PRED = getattr(settings, 'RUTA_DATOS_MODELO', '')

# ----------------- VISTAS TEMPLATES -----------------
# -------------------- Función para cargar métricas --------------------
def load_metrics(target):
    """Carga métricas desde CSV y mapea los nombres a los esperados por el HTML"""
    metrics_path = os.path.join(settings.OUT_DIR, f"metrics_{target}.csv")
    if not os.path.exists(metrics_path):
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    # Tomar la primera fila
    row = df.iloc[0].to_dict()

    # Mapeo de nombres CSV -> nombres esperados en HTML
    mapping = {
        "MAE_lin": "MAE_test_lin",
        "RMSE_lin": "RMSE_test_lin",
        "R2_lin": "R2_test_lin",
        "MAE_rf": "MAE_test_rf",
        "RMSE_rf": "RMSE_test_rf",
        "R2_rf": "R2_test_rf"
    }

    mapped_row = {}
    for k, v in mapping.items():
        try:
            mapped_row[v] = float(row[k])
        except:
            mapped_row[v] = row.get(k, "N/A")

    return SimpleNamespace(**mapped_row)

# -------------------- Vista del dashboard --------------------
def view_ct_analisis_home(request):
    """Pantalla inicial del módulo de análisis de datos"""
    graficas = {}
    
    for TARGET in ["PM2_5", "CO2"]:
        csv_path = os.path.join(settings.OUT_DIR, f"predicciones_{TARGET}.csv")
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df["datetime"] = pd.to_datetime(df["datetime"])
        except Exception as e:
            graficas[TARGET] = f"<p>Error leyendo CSV ({TARGET}): {e}</p>"
            continue

        # Graficar predicciones
        if TARGET == "PM2_5":
            y_real_col = "pm25_valor_real"
            y_lin_col = "pm25_pred_reg_lineal"
            y_rf_col = "pm25_pred_randomforest"
        else:
            y_real_col = "co2_valor_real"
            y_lin_col = "co2_pred_reg_lineal"
            y_rf_col = "co2_pred_randomforest"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_real_col], name="Valor real", line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_lin_col], name="Regresión Lineal", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df["datetime"], y=df[y_rf_col], name="Random Forest"))
        fig.update_layout(title=f"Comparación Real vs Predicciones ({TARGET})",
                          xaxis_title="Fecha", yaxis_title=TARGET, hovermode="x unified")
        graficas[TARGET] = fig.to_html(full_html=False)

    # Cargar métricas
    metrics_pm25 = load_metrics("PM2_5")
    metrics_co2 = load_metrics("CO2")

    # Si no se encuentran métricas, crear SimpleNamespace con "N/A"
    if metrics_pm25 is None:
        metrics_pm25 = SimpleNamespace(
            MAE_test_lin="N/A", RMSE_test_lin="N/A", R2_test_lin="N/A",
            MAE_test_rf="N/A", RMSE_test_rf="N/A", R2_test_rf="N/A"
        )
    if metrics_co2 is None:
        metrics_co2 = SimpleNamespace(
            MAE_test_lin="N/A", RMSE_test_lin="N/A", R2_test_lin="N/A",
            MAE_test_rf="N/A", RMSE_test_rf="N/A", R2_test_rf="N/A"
        )

    # Contexto para template
    ctx = {
        "grafica_pm25": graficas.get("PM2_5", ""),
        "grafica_co2": graficas.get("CO2", ""),
        "metrics_pm25": metrics_pm25,
        "metrics_co2": metrics_co2,
    }

    return render(request, 'analisis_rl_rf/analisis_home.html', ctx)

## Esta funcion se encarga de mostrar las dos imagenes generadas 
def view_ct_resultados_modelos(request):
    """Mostrar las imágenes generadas por los modelos"""

    # Rutas a las imágenes generadas
    pm_rf_png = static("dashboard/pm25_rf.png")
    pm_lin_png = static("dashboard/pm25_lin.png")
    co2_rf_png = static("dashboard/co2_rf.png")
    co2_lin_png = static("dashboard/co2_lin.png")

    pm_data = {
        'RF': {'png_url': pm_rf_png},
        'Linear': {'png_url': pm_lin_png}
    }

    co2_data = {
        'RF': {'png_url': co2_rf_png},
        'Linear': {'png_url': co2_lin_png}
    }

    ctx = {
        'pm_data': pm_data,
        'co2_data': co2_data
    }

    return render(request, 'analisis_rl_rf/resultados_modelos.html', ctx)




def view_ct_tabla_predicciones(request):
    """Resumen general de resultados usando datos de CSV de predicciones"""
    df = None
    table_rows = []
    chart_labels = []
    chart_real = []
    chart_rf = []
    error_msg = None

    FRIENDLY_COLUMNS = {
        "datetime": "Fecha y hora",
        "pm25_valor_real": "PM2.5 real (µg/m³)",
        "pm25_pred_reg_lineal": "Predicción PM2.5 – Regresión Lineal",
        "pm25_pred_randomforest": "Predicción PM2.5 – Random Forest",
        "co2_valor_real": "CO₂ real (ppm)",
        "co2_pred_reg_lineal": "Predicción CO₂ – Regresión Lineal",
        "co2_pred_randomforest": "Predicción CO₂ – Random Forest",
    }

    try:
        if os.path.exists(CSV_PRED):
            df = pd.read_csv(CSV_PRED)
            df.columns = df.columns.str.strip().str.lower()
            cols = df.columns.tolist()

            if 'datetime' in cols:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

            table_rows = df.to_dict(orient='records')
            chart_labels = df['datetime'].tolist() if 'datetime' in cols else []

            col_true = 'pm25_valor_real'
            col_rf = 'pm25_pred_randomforest'
            if col_true in cols and col_rf in cols:
                chart_real = df[col_true].tolist()
                chart_rf = df[col_rf].tolist()
            else:
                error_msg = "El CSV no contiene las columnas necesarias para la gráfica (pm25_valor_real, pm25_pred_randomforest)."
        else:
            error_msg = f"No se encontró el archivo CSV: {CSV_PRED}"

    except Exception as e:
        error_msg = f"Error leyendo CSV: {e}"

    # Encabezados amigables
    friendly_headers = [FRIENDLY_COLUMNS.get(col, col) for col in df.columns] if isinstance(df, pd.DataFrame) else []

    ctx = {
        'error_msg': error_msg,
        'table_rows': table_rows,
        'columns': friendly_headers,

        # Gráfica
        'chart_labels': chart_labels,
        'chart_real': chart_real,
        'chart_rf': chart_rf,
    }

    return render(request, 'analisis_rl_rf/tabla_predicciones.html', ctx)

