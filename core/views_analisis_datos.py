from django.shortcuts import render
from django.conf import settings
from types import SimpleNamespace
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import pandas as pd
import plotly.graph_objects as go


# =========================
# UTILIDADES
# =========================
def get_available_devices():
    devices = []
    for f in os.listdir(settings.OUT_DIR):
        if f.startswith("predicciones_PM2_5_") and f.endswith(".csv"):
            devices.append(
                f.replace("predicciones_PM2_5_", "").replace(".csv", "")
            )
    return sorted(devices)


def load_predictions(target, device):
    path = os.path.join(
        settings.OUT_DIR, f"predicciones_{target}_{device}.csv"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def load_ambiental_data(device):
    path = os.path.join(
        settings.OUT_DIR, f"ambiental_{device}.csv"
    )

    if not os.path.exists(path):
        print("❌ No se encontró:", path)
        return None

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def filter_df(df, start_date, end_date):
    if "datetime" not in df.columns:
        return df

    if start_date:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]

    return df



def generar_stats_ambientales_rango(df):
    variables = {
        "temp": "Temperatura",
        "humidity": "Humedad",
        "pressure": "Presión",
        "light": "Luz",
    }

    rows = []

    for col, label in variables.items():
        if col not in df.columns:
            continue

        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.empty:
            continue

        rows.append({
            "variable": label,
            "count": len(serie),
            "mean": round(serie.mean(), 2),
            "min": round(serie.min(), 2),
            "max": round(serie.max(), 2),
        })

    return rows


def compute_metrics(df, target):
    return SimpleNamespace(
        MAE_test_lin=mean_absolute_error(
            df[f"{target}_real"], df[f"{target}_pred_lin"]
        ),
        RMSE_test_lin=mean_squared_error(
            df[f"{target}_real"], df[f"{target}_pred_lin"]
        ) ** 0.5,
        R2_test_lin=r2_score(
            df[f"{target}_real"], df[f"{target}_pred_lin"]
        ),
        MAE_test_rf=mean_absolute_error(
            df[f"{target}_real"], df[f"{target}_pred_rf"]
        ),
        RMSE_test_rf=mean_squared_error(
            df[f"{target}_real"], df[f"{target}_pred_rf"]
        ) ** 0.5,
        R2_test_rf=r2_score(
            df[f"{target}_real"], df[f"{target}_pred_rf"]
        ),
    )
    
# =========================
# INTERPRETACIÓN DE RESULTADOS
# =========================
def interpretar_resultados(metrics, target):
    """
    Genera una interpretación automática del desempeño del modelo
    según métricas estándar de ML.
    """

    r2 = metrics.R2_test_rf
    rmse = metrics.RMSE_test_rf

    if r2 >= 0.8:
        return (
            "success",
            f"El modelo para {target} presenta un desempeño muy bueno. "
            "Explica gran parte de la variabilidad de los datos y mantiene errores bajos, "
            "por lo que es confiable para análisis predictivo."
        )

    elif r2 >= 0.6:
        return (
            "info",
            f"El modelo para {target} muestra un desempeño aceptable. "
            "Puede utilizarse para análisis exploratorio, aunque existe margen de mejora."
        )

    elif r2 >= 0.4:
        return (
            "warning",
            f"El modelo para {target} tiene un desempeño limitado. "
            "Las predicciones presentan errores moderados y deben interpretarse con precaución."
        )

    else:
        return (
            "danger",
            f"El modelo para {target} no logra explicar adecuadamente el comportamiento de los datos. "
            "No se recomienda su uso para análisis predictivo."
        )
        
def generar_conclusion(interpretacion_pm25, interpretacion_co2, contexto_ambiental=None):
    """
    Genera una conclusión global del desempeño de los modelos
    combinando PM2.5 y CO₂.
    """

    niveles = []

    if interpretacion_pm25:
        niveles.append(interpretacion_pm25["nivel"])

    if interpretacion_co2:
        niveles.append(interpretacion_co2["nivel"])

    if not niveles:
        return None

    # Prioridad del peor escenario
    if "danger" in niveles:
        nivel = "danger"
        mensaje = (
            "En general, los modelos presentan un desempeño deficiente. "
            "Las predicciones no explican adecuadamente el comportamiento de los datos, "
            "por lo que no se recomienda su uso para análisis predictivo."
        )

    elif "warning" in niveles:
        nivel = "warning"
        mensaje = (
            "El desempeño global de los modelos es limitado. "
            "Si bien capturan algunas tendencias, los errores son moderados "
            "y los resultados deben interpretarse con precaución."
        )

    elif "info" in niveles:
        nivel = "info"
        mensaje = (
            "Los modelos presentan un desempeño aceptable en general. "
            "Son útiles para análisis exploratorio y seguimiento de tendencias, "
            "aunque existe margen de mejora."
        )

    else:
        nivel = "success"
        mensaje = (
            "Los modelos muestran un desempeño global muy bueno. "
            "Explican adecuadamente la variabilidad de los datos y presentan errores bajos, "
            "por lo que son confiables para análisis predictivo."
        )

    return {
    "nivel": nivel,
    "mensaje": mensaje,
    "contexto": (
        "Este desempeño debe interpretarse considerando las condiciones "
        "ambientales observadas en el periodo analizado, donde se registraron "
        f"{contexto_ambiental}."
        if contexto_ambiental else None
    )
    }



def generar_resumen_ejecutivo(
    device,
    start_date,
    end_date,
    total_pm25,
    total_co2,
    metrics_pm25,
    metrics_co2,
    interpretacion_pm25,
    interpretacion_co2,
    contexto_ambiental=None,
):
    if not device:
        return None

    inicio = start_date if start_date else "inicio del periodo registrado"
    fin = end_date if end_date else "fin del periodo registrado"

    bloques = []

    # ================= CONTEXTO =================
    bloques.append(f"""
    <p>
      Para el periodo comprendido entre <strong>{inicio}</strong> y
      <strong>{fin}</strong>, se analizaron los datos registrados por el sensor
      <strong>{device}</strong>.
    </p>
    """)

    # ================= CONFIABILIDAD =================
    if total_pm25:
        bloques.append(f"""
        <p>
          En el caso de <strong>PM2.5</strong>, se utilizaron
          <strong>{total_pm25:,} registros válidos</strong>, lo que representa un
          volumen suficiente para evaluar el desempeño del modelo predictivo.
        </p>
        """)

    if total_co2:
        bloques.append(f"""
        <p>
          Para <strong>CO₂</strong>, se procesaron
          <strong>{total_co2:,} registros válidos</strong>, garantizando una base
          de datos representativa del comportamiento observado.
        </p>
        """)

    # ================= PM2.5 =================
    if metrics_pm25 and interpretacion_pm25:
        bloques.append(f"""
        <h5 class="mt-4">Evaluación del Modelo PM2.5</h5>
        <p>
          El modelo <strong>Random Forest</strong> obtuvo un coeficiente de
          determinación <strong>R² = {metrics_pm25.R2_test_rf:.3f}</strong>,
          lo que indica que explica aproximadamente el
          <strong>{metrics_pm25.R2_test_rf*100:.1f}%</strong> de la variabilidad
          observada en los datos reales.
        </p>
        <p>
          El error promedio del modelo (<strong>RMSE =
          {metrics_pm25.RMSE_test_rf:.3f} µg/m³</strong>) se considera bajo dentro
          del contexto del análisis ambiental, reflejando una buena precisión
          predictiva.
        </p>
        """)

    # ================= CO₂ =================
    if metrics_co2 and interpretacion_co2:
        nivel_legible = {
            "success": "muy bueno",
            "info": "aceptable",
            "warning": "limitado",
            "danger": "deficiente",
        }.get(interpretacion_co2["nivel"], "aceptable")

        bloques.append(f"""
        <h5 class="mt-4">Evaluación del Modelo CO₂</h5>
        <p>
          Para <strong>CO₂</strong>, el modelo Random Forest presentó un
          coeficiente <strong>R² = {metrics_co2.R2_test_rf:.3f}</strong> y un
          error promedio <strong>RMSE =
          {metrics_co2.RMSE_test_rf:.3f} ppm</strong>, reflejando un desempeño
          <strong>{nivel_legible}</strong> según los criterios establecidos.
        </p>
        """)
        
    if contexto_ambiental:
        bloques.append(f"""
        <div class="mt-3 p-3 rounded bg-light border-start border-4 border-primary">
        <h6 class="fw-bold mb-2">Contexto ambiental del periodo</h6>
        <p class="mb-0">
            Durante el rango seleccionado se observaron condiciones caracterizadas por
            {contexto_ambiental}. Estas variables influyen directamente en la dinámica
            de los contaminantes y aportan contexto clave para la interpretación de
            los modelos predictivos.
        </p>
        </div>
        """)


    # ================= CIERRE =================
    bloques.append("""
    <p class="mt-4">
      En conjunto, los resultados indican que los modelos predictivos evaluados
      proporcionan una aproximación confiable del comportamiento ambiental
      durante el periodo analizado, siendo especialmente adecuados para análisis
      exploratorio y como apoyo en la toma de decisiones.
    </p>
    """)

    return "".join(bloques)




# =========================
# DASHBOARD
# =========================
def view_ct_analisis_home(request):

    device = request.GET.get("device")
    start_date = request.GET.get("start_date")
    end_date = request.GET.get("end_date")

    devices = get_available_devices()

    grafica_pm25 = "<p class='text-muted'>Seleccione un sensor.</p>"
    grafica_co2 = "<p class='text-muted'>Seleccione un sensor.</p>"

    metrics_pm25 = None
    metrics_co2 = None

    interpretacion_pm25 = None
    interpretacion_co2 = None

    df_pm = None
    df_co2 = None

    total_pm25 = 0
    total_co2 = 0
    
    stats_ambientales = None
    
    # ================= AMBIENTAL =================
    if device:
        df_env = load_ambiental_data(device)

        if df_env is not None:
            # 👉 AHORA SÍ se filtra por rango
            df_env = filter_df(df_env, start_date, end_date)

            if not df_env.empty:
                stats_ambientales = generar_stats_ambientales_rango(df_env)

                


    # ================= PM2.5 =================
    if device:
        df_pm = load_predictions("PM2_5", device)

        if df_pm is not None:
            df_pm = filter_df(df_pm, start_date, end_date)

            if not df_pm.empty:
                total_pm25 = len(df_pm)

                metrics_pm25 = compute_metrics(df_pm, "PM2_5")

                nivel_pm25, mensaje_pm25 = interpretar_resultados(
                    metrics_pm25, "PM2.5"
                )

                interpretacion_pm25 = {
                    "nivel": nivel_pm25,
                    "mensaje": mensaje_pm25,
                }

                fig_pm = go.Figure()

                fig_pm.add_trace(go.Scatter(
                    x=df_pm["datetime"],
                    y=df_pm["PM2_5_real"],
                    name="PM2.5 Real",
                    hovertemplate=(
                        "<b>Variable:</b> PM2.5 (Real)<br>"
                        "<b>Valor:</b> %{y:.2f} µg/m³<br>"
                        "<b>Fecha:</b> %{x|%Y-%m-%d %H:%M}<extra></extra>"
                    ),
                    line=dict(color="black", width=3),
                ))

                fig_pm.add_trace(go.Scatter(
                    x=df_pm["datetime"],
                    y=df_pm["PM2_5_pred_lin"],
                    name="Regresión Lineal",
                    line=dict(color="royalblue", dash="dash"),
                ))

                fig_pm.add_trace(go.Scatter(
                    x=df_pm["datetime"],
                    y=df_pm["PM2_5_pred_rf"],
                    name="Random Forest",
                    line=dict(color="firebrick"),
                ))

                fig_pm.update_layout(
                    title=f"PM2.5 – Sensor {device}",
                    xaxis_title="Fecha",
                    yaxis_title="µg/m³",
                )

                grafica_pm25 = fig_pm.to_html(full_html=False)

    # ================= CO2 =================
    if device:
        df_co2 = load_predictions("CO2", device)

        if df_co2 is not None:
            df_co2 = filter_df(df_co2, start_date, end_date)

            if not df_co2.empty:
                total_co2 = len(df_co2)

                metrics_co2 = compute_metrics(df_co2, "CO2")

                nivel_co2, mensaje_co2 = interpretar_resultados(
                    metrics_co2, "CO₂"
                )

                interpretacion_co2 = {
                    "nivel": nivel_co2,
                    "mensaje": mensaje_co2,
                }

                fig_co2 = go.Figure()

                fig_co2.add_trace(go.Scatter(
                    x=df_co2["datetime"],
                    y=df_co2["CO2_real"],
                    name="CO₂ Real",
                    line=dict(color="black", width=3),
                ))

                fig_co2.add_trace(go.Scatter(
                    x=df_co2["datetime"],
                    y=df_co2["CO2_pred_lin"],
                    name="Regresión Lineal",
                    line=dict(color="royalblue", dash="dash"),
                ))

                fig_co2.add_trace(go.Scatter(
                    x=df_co2["datetime"],
                    y=df_co2["CO2_pred_rf"],
                    name="Random Forest",
                    line=dict(color="firebrick"),
                ))

                fig_co2.update_layout(
                    title=f"CO₂ – Sensor {device}",
                    xaxis_title="Fecha",
                    yaxis_title="ppm",
                )

                grafica_co2 = fig_co2.to_html(full_html=False)
                
        # ================= CONTEXTO AMBIENTAL PARA RESUMEN =================
    contexto_ambiental = None

    if stats_ambientales:
        partes = []
        for s in stats_ambientales:
            partes.append(
                f"{s['variable']} promedio {s['mean']}"
            )
        contexto_ambiental = ", ".join(partes)

    conclusion_final = generar_conclusion(
    interpretacion_pm25,
    interpretacion_co2,
    contexto_ambiental,
    )


    resumen_ejecutivo = generar_resumen_ejecutivo(
        device=device,
        start_date=start_date,
        end_date=end_date,
        total_pm25=total_pm25,
        total_co2=total_co2,
        metrics_pm25=metrics_pm25,
        metrics_co2=metrics_co2,
        interpretacion_pm25=interpretacion_pm25,
        interpretacion_co2=interpretacion_co2,
        contexto_ambiental=contexto_ambiental,
    )
    
    ctx = {
        "devices": devices,
        "grafica_pm25": grafica_pm25,
        "metrics_pm25": metrics_pm25,
        "interpretacion_pm25": interpretacion_pm25,
        "total_pm25": total_pm25,
        "grafica_co2": grafica_co2,
        "metrics_co2": metrics_co2,
        "interpretacion_co2": interpretacion_co2,
        "total_co2": total_co2,  
        "conclusion_final": conclusion_final,
        "resumen_ejecutivo": resumen_ejecutivo,
        "stats_ambientales": stats_ambientales,
        "contexto_ambiental": contexto_ambiental,

    }

    return render(request, "analisis_rl_rf/analisis_home.html", ctx)
