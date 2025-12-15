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
    Dashboard PM2.5 usando datos desde MongoDB
    """

    # ---- Datos reales desde Mongo ----
    df_real = get_mongo_df_pm25()

    # ---- Predicciones desde OUT_DIR (por ahora) ----
    pred_path = os.path.join(settings.OUT_DIR, "predicciones_PM2_5.csv")

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])

        df = pd.merge(
            df_real,
            df_pred,
            on="datetime",
            how="inner"
        )
    else:
        df = df_real.copy()
        df["pred_regresion_lineal"] = None
        df["pred_random_forest"] = None

    # ---- Gráfica ----
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["pm25_real"],
        name="PM2.5 Real",
        line=dict(color="black", width=3)
    ))

    if "pred_regresion_lineal" in df:
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df["pred_regresion_lineal"],
            name="Regresión Lineal",
            line=dict(dash="dash")
        ))

    if "pred_random_forest" in df:
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df["pred_random_forest"],
            name="Random Forest"
        ))

    fig.update_layout(
        title="PM2.5 – Datos reales vs predicciones",
        xaxis_title="Fecha",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode="x unified"
    )

    grafica_pm25 = fig.to_html(full_html=False)

    # ---- Métricas ----
    metrics_path = os.path.join(settings.OUT_DIR, "metrics_PM2_5.csv")

    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path).iloc[0]
        metrics_pm25 = SimpleNamespace(
            MAE_test_lin=round(m["MAE_lin"], 3),
            RMSE_test_lin=round(m["RMSE_lin"], 3),
            R2_test_lin=round(m["R2_lin"], 3),
            MAE_test_rf=round(m["MAE_rf"], 3),
            RMSE_test_rf=round(m["RMSE_rf"], 3),
            R2_test_rf=round(m["R2_rf"], 3),
        )
    else:
        metrics_pm25 = SimpleNamespace(
            MAE_test_lin="N/A",
            RMSE_test_lin="N/A",
            R2_test_lin="N/A",
            MAE_test_rf="N/A",
            RMSE_test_rf="N/A",
            R2_test_rf="N/A",
        )

    ctx = {
        "grafica_pm25": grafica_pm25,
        "metrics_pm25": metrics_pm25,
        "grafica_co2": "",      # para que no rompa el template
        "metrics_co2": None,
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



def get_pm25_from_mongo(limit=1000):
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
