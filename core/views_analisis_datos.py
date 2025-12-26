from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.templatetags.static import static
from django.core.management import call_command
from types import SimpleNamespace
from pymongo import MongoClient

import os
import pandas as pd
import plotly.graph_objects as go

CSV_PRED = getattr(settings, 'RUTA_DATOS_MODELO', '')

# =========================
# CONEXIÓN A MONGO
# =========================
def get_mongo_df_pm25():
    client = MongoClient(
        host="159.203.44.80",
        port=27017,
        username="CityAirLogs-DjangoSystem",
        password="0081KCW=vyvaneLe@9677-3659_sm!:GMCDJ-ñ",
        authSource="admin"
    )

    db = client["CityAirLogs"]
    collection = db["lecturas"]

    cursor = collection.find(
        {},
        {
            "_id": 0,
            "fecha": 1,
            "PM2_5": 1
        }
    )

    df = pd.DataFrame(list(cursor))
    df["datetime"] = pd.to_datetime(df["fecha"], errors="coerce")
    df.rename(columns={"PM2_5": "pm25_real"}, inplace=True)
    df = df.dropna(subset=["datetime", "pm25_real"])

    return df.sort_values("datetime")


# =========================
# DASHBOARD
# =========================
def view_ct_analisis_home(request):
    """
    Dashboard PM2.5 y CO₂
    usando únicamente resultados generados por los modelos
    """

    # =========================
    # PM2.5
    # =========================
    pm_pred_path = os.path.join(settings.OUT_DIR, "predicciones_PM2_5.csv")
    pm_metrics_path = os.path.join(settings.OUT_DIR, "metrics_PM2_5.csv")

    grafica_pm25 = "<p>No hay datos de PM2.5 aún.</p>"
    metrics_pm25 = None

    if os.path.exists(pm_pred_path):
        df_pm = pd.read_csv(pm_pred_path)
        df_pm["datetime"] = pd.to_datetime(df_pm["datetime"])

        fig_pm = go.Figure()

        fig_pm.add_trace(go.Scatter(
            x=df_pm["datetime"],
            y=df_pm["PM2_5_real"],
            name="PM2.5 Real",
            line=dict(color="black", width=3)
        ))

        fig_pm.add_trace(go.Scatter(
            x=df_pm["datetime"],
            y=df_pm["PM2_5_pred_lin"],
            name="Regresión Lineal",
            line=dict(dash="dash")
        ))

        fig_pm.add_trace(go.Scatter(
            x=df_pm["datetime"],
            y=df_pm["PM2_5_pred_rf"],
            name="Random Forest"
        ))

        fig_pm.update_layout(
            title="PM2.5 – Datos reales vs predicciones",
            xaxis_title="Fecha",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified"
        )

        grafica_pm25 = fig_pm.to_html(full_html=False)

    if os.path.exists(pm_metrics_path):
        m = pd.read_csv(pm_metrics_path).iloc[0]
        metrics_pm25 = SimpleNamespace(
            MAE_test_lin=round(m["MAE_lin"], 3),
            RMSE_test_lin=round(m["RMSE_lin"], 3),
            R2_test_lin=round(m["R2_lin"], 3),
            MAE_test_rf=round(m["MAE_rf"], 3),
            RMSE_test_rf=round(m["RMSE_rf"], 3),
            R2_test_rf=round(m["R2_rf"], 3),
        )

    # =========================
    # CO₂
    # =========================
    co2_pred_path = os.path.join(settings.OUT_DIR, "predicciones_CO2.csv")
    co2_metrics_path = os.path.join(settings.OUT_DIR, "metrics_CO2.csv")

    grafica_co2 = "<p>No hay datos de CO₂ aún.</p>"
    metrics_co2 = None

    if os.path.exists(co2_pred_path):
        df_co2 = pd.read_csv(co2_pred_path)
        df_co2["datetime"] = pd.to_datetime(df_co2["datetime"])

        fig_co2 = go.Figure()

        fig_co2.add_trace(go.Scatter(
            x=df_co2["datetime"],
            y=df_co2["CO2_real"],
            name="CO₂ Real",
            line=dict(color="black", width=3)
        ))

        fig_co2.add_trace(go.Scatter(
            x=df_co2["datetime"],
            y=df_co2["CO2_pred_lin"],
            name="Regresión Lineal",
            line=dict(dash="dash")
        ))

        fig_co2.add_trace(go.Scatter(
            x=df_co2["datetime"],
            y=df_co2["CO2_pred_rf"],
            name="Random Forest"
        ))

        fig_co2.update_layout(
            title="CO₂ – Datos reales vs predicciones",
            xaxis_title="Fecha",
            yaxis_title="CO₂ (ppm)",
            hovermode="x unified"
        )

        grafica_co2 = fig_co2.to_html(full_html=False)

    if os.path.exists(co2_metrics_path):
        m = pd.read_csv(co2_metrics_path).iloc[0]
        metrics_co2 = SimpleNamespace(
            MAE_test_lin=round(m["MAE_lin"], 3),
            RMSE_test_lin=round(m["RMSE_lin"], 3),
            R2_test_lin=round(m["R2_lin"], 3),
            MAE_test_rf=round(m["MAE_rf"], 3),
            RMSE_test_rf=round(m["RMSE_rf"], 3),
            R2_test_rf=round(m["R2_rf"], 3),
        )

    # =========================
    # CONTEXTO
    # =========================
    ctx = {
        "grafica_pm25": grafica_pm25,
        "metrics_pm25": metrics_pm25,
        "grafica_co2": grafica_co2,
        "metrics_co2": metrics_co2,
    }

    return render(request, "analisis_rl_rf/analisis_home.html", ctx)



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


## TABLA DE DATOS



# def get_pm25_from_mongo(limit=1000):
    client = MongoClient(settings.MONGO_URI)
    db = client["CityAirLogs"]
    col = db["lecturas"]

    cursor = (
        col.find(
            {"pm25": {"$exists": True}},
            {"_id": 0}
        )
        .sort("datetime", -1)   # más recientes primero
        .limit(limit)
    )

    df = pd.DataFrame(list(cursor))
    client.close()

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def view_ct_tabla_predicciones(request):
    table_rows = []
    error_msg = None

    FRIENDLY_COLUMNS = {
        "datetime": "Fecha y hora",
        "pm25_valor_real": "PM2.5 real (µg/m³)",
        "pm25_pred_reg_lineal": "Predicción PM2.5 – Regresión Lineal",
        "pm25_pred_randomforest": "Predicción PM2.5 – Random Forest",
    }

    try:
        df = get_pm25_from_mongo(limit=1000)

        if df.empty:
            error_msg = "No hay datos en MongoDB."
        else:
            df = df.sort_values("datetime")
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
            table_rows = df.to_dict(orient="records")

    except Exception as e:
        error_msg = f"Error leyendo MongoDB: {e}"

    columns = [FRIENDLY_COLUMNS.get(c, c) for c in df.columns] if not df.empty else []

    ctx = {
        "error_msg": error_msg,
        "table_rows": table_rows,
        "columns": columns,
    }

    return render(request, "analisis_rl_rf/tabla_predicciones.html", ctx)
