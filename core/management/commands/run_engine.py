import math
from datetime import datetime

import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ======================================================
# FEATURES DISPONIBLES
# ======================================================
def get_available_features(df):
    base = ["temp", "humidity", "pressure", "light"]
    return [c for c in base if c in df.columns]


# ======================================================
# ENTRENAMIENTO
# ======================================================
def entrenar_modelo(df, target):
    """
    Entrena modelo en memoria de forma segura.
    Crea variables temporales antes de limpiar.
    """

    if "datetime" not in df.columns:
        return None, None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # variables temporales
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek

    base_features = ["temp", "humidity", "pressure", "light"]
    env_features = [f for f in base_features if f in df.columns]
    time_features = ["hour", "dow"]
    features = env_features + time_features

    if not env_features:
        return None, None

    # -------------------------
    # 3. Limpieza segura
    # -------------------------

    required_cols = ["datetime", target] + features

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas para {target}: {missing}")
        return None, None

    df = df.dropna(subset=required_cols)

    if len(df) < 50:
        return None, None

    # -------------------------
    # 4. Preparación X, y
    # -------------------------

    X = df[features]
    y = df[target]

    # -------------------------
    # 5. Pipelines
    # -------------------------

    pre = ColumnTransformer([("num", StandardScaler(), features)])

    rf = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )

    # -------------------------
    # 6. Entrenamiento
    # -------------------------

    rf.fit(X, y)
    yhat = rf.predict(X)

    # -------------------------
    # 6. Métricas
    # -------------------------

    metrics = {
        "features": features,
        "R2": r2_score(y, yhat),
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": math.sqrt(mean_squared_error(y, yhat)),
        "n_samples": len(df),
    }

    # -------------------------
    # 7. Predicciones
    # -------------------------

    preds = pd.DataFrame(
        {
            "datetime": df["datetime"],
            "y_true": y,
            "y_pred": yhat,
        }
    ).sort_values("datetime")

    return metrics, preds, rf


# ======================================================
# GENERACIÓN DE FEATURES FUTURAS (ROBUSTA)
# ======================================================
def generar_features_futuras(df, steps=48):
    """
    Genera features futuras de forma segura para predicción.
    Usa persistencia + variables temporales.
    """

    if df is None or df.empty:
        print("⚠️ generar_features_futuras: DF vacío")
        return None

    if "datetime" not in df.columns:
        print("⚠️ generar_features_futuras: no existe datetime")
        return None

    df = df.sort_values("datetime")

    last_row = df.iloc[-1]

    # Validar que exista al menos una variable ambiental
    env_cols = [c for c in ["temp", "humidity", "pressure", "light"] if c in df.columns]
    if not env_cols:
        print("⚠️ generar_features_futuras: no hay columnas ambientales")
        return None

    # Fechas futuras (usar 'h' para evitar warning)
    future_dates = pd.date_range(
        start=last_row["datetime"] + pd.Timedelta(hours=1),
        periods=steps,
        freq="h",
    )

    future = pd.DataFrame({"datetime": future_dates})

    # Variables temporales
    future["hour"] = future["datetime"].dt.hour
    future["dow"] = future["datetime"].dt.dayofweek

    # Persistencia ambiental
    for col in env_cols:
        val = last_row[col]
        if pd.isna(val):
            # usar promedio si último valor es NaN
            val = df[col].dropna().mean()
        future[col] = val

    print("✅ generar_features_futuras OK:", future.head(2).to_dict())

    return future


# ======================================================
# PREDICCIÓN FUTURA CON MODELO ML
# ======================================================
def predecir_futuro(model, df_features):
    """
    Usa el pipeline entrenado para predecir valores futuros.
    """
    if model is None or df_features is None or df_features.empty:
        return None

    try:
        X_future = df_features.drop(columns=["datetime"])
        y_future = model.predict(X_future)

        out = df_features.copy()
        out["y_pred"] = y_future
        return out
    except Exception as e:
        print("❌ predecir_futuro error:", e)
        return None


# ======================================================
# PREDICCIÓN FUTURA SIMPLE (ROBUSTA)
# ======================================================
def predecir_futuro_simple(df, target, steps=48):
    """
    Genera predicción futura usando persistencia del promedio reciente.
    NO depende del modelo ML.
    Siempre devuelve resultado.
    """
    if df is None or df.empty:
        return None

    if target not in df.columns or "datetime" not in df.columns:
        return None

    df = df.dropna(subset=[target, "datetime"]).sort_values("datetime")

    if len(df) < 10:
        return None

    # Promedio de las últimas 24 horas (o últimos registros)
    base_val = df[target].tail(24).mean()

    last_dt = df["datetime"].iloc[-1]

    future_dates = pd.date_range(
        start=last_dt + pd.Timedelta(hours=1),
        periods=steps,
        freq="h",
    )

    future = pd.DataFrame(
        {
            "datetime": future_dates,
            "y_pred": [base_val] * steps,
        }
    )

    print(f"✅ prediccion futura simple OK para {target}: {base_val:.2f}")

    return future


# ======================================================
# CLASIFICACIÓN PM2.5
# ======================================================
def _clasificar_pm25(pm25_mean):
    if pm25_mean <= 15:
        return "BUENO", "🟢", "El aire es adecuado para actividades al aire libre."
    elif pm25_mean <= 33:
        return (
            "ACEPTABLE",
            "🟡",
            (
                "La calidad del aire es moderada. "
                "Personas sensibles deben evitar esfuerzos prolongados."
            ),
        )
    elif pm25_mean <= 55:
        return (
            "MALO",
            "🟠",
            (
                "La calidad del aire es poco saludable. "
                "Se recomienda reducir actividades en exteriores."
            ),
        )
    else:
        return (
            "MUY MALO",
            "🔴",
            (
                "La calidad del aire es peligrosa. "
                "Se aconseja permanecer en interiores."
            ),
        )


# ======================================================
# MENSAJES PARA DASHBOARD (PREDICCIÓN FUTURA)
# ======================================================
def _nivel_bootstrap_pm25(valor):
    """
    Devuelve clase Bootstrap según nivel de PM2.5
    """
    if valor <= 15:
        return "success"
    elif valor <= 33:
        return "warning"
    elif valor <= 55:
        return "danger"
    else:
        return "danger"


def generar_texto_dashboard(future_df, contaminante="PM2_5"):
    """
    Genera mensajes con tono:
    científico + claro + cercano
    """
    if future_df is None or future_df.empty:
        return None

    mean_val = future_df["y_pred"].mean()
    max_val = future_df["y_pred"].max()

    # ---------------------------
    # PM2.5
    # ---------------------------
    if contaminante == "PM2_5":

        # Nivel visual
        if mean_val <= 15:
            nivel_css = "success"
            nivel_txt = "bueno"
        elif mean_val <= 33:
            nivel_css = "warning"
            nivel_txt = "moderado"
        elif mean_val <= 55:
            nivel_css = "danger"
            nivel_txt = "alto"
        else:
            nivel_css = "danger"
            nivel_txt = "muy alto"

        mensaje = (
            f"Con base en el análisis de los <strong>registros históricos más recientes</strong>, "
            f"el modelo de predicción estima que la concentración de <strong>PM2.5</strong> "
            f"se mantendrá en un nivel <strong>{nivel_txt}</strong> durante las próximas "
            f"<strong>48 horas</strong>. "
            f"El valor promedio esperado es de aproximadamente "
            f"<strong>{mean_val:.1f} µg/m³</strong>. "
        )

        if nivel_txt in ["bueno"]:
            mensaje += (
                "En este escenario, la calidad del aire es adecuada para la mayoría de las "
                "actividades diarias y no se esperan efectos adversos en la salud."
            )
        elif nivel_txt in ["moderado"]:
            mensaje += (
                "Aunque este nivel no representa un riesgo grave, "
                "las personas con asma, alergias o sensibilidad respiratoria "
                "podrían experimentar molestias leves si realizan actividades "
                "prolongadas al aire libre."
            )
        elif nivel_txt in ["alto"]:
            mensaje += (
                "Este nivel puede generar molestias en grupos vulnerables, "
                "por lo que se recomienda reducir el tiempo de exposición "
                "en exteriores, especialmente en horarios de mayor concentración."
            )
        else:
            mensaje += (
                "Este escenario indica una posible afectación a la salud general. "
                "Se aconseja permanecer en espacios cerrados y limitar actividades "
                "físicas al aire libre."
            )

    # ---------------------------
    # CO2
    # ---------------------------
    else:

        if mean_val <= 800:
            nivel_css = "success"
            nivel_txt = "normal"
        elif mean_val <= 1200:
            nivel_css = "warning"
            nivel_txt = "moderado"
        else:
            nivel_css = "danger"
            nivel_txt = "elevado"

        mensaje = (
            f"A partir de los datos recopilados recientemente, "
            f"el sistema estima que los niveles de <strong>CO₂</strong> "
            f"se mantendrán en un rango <strong>{nivel_txt}</strong> "
            f"durante las próximas <strong>48 horas</strong>, "
            f"con un valor promedio cercano a "
            f"<strong>{mean_val:.0f} ppm</strong>. "
        )

        if nivel_txt == "normal":
            mensaje += (
                "Estas concentraciones son adecuadas para espacios interiores "
                "y no se asocian con efectos negativos en el bienestar general."
            )
        elif nivel_txt == "moderado":
            mensaje += (
                "Aunque no representan un riesgo inmediato, "
                "estos niveles pueden generar sensación de aire cargado, "
                "por lo que se recomienda mantener una buena ventilación."
            )
        else:
            mensaje += (
                "Concentraciones elevadas de CO₂ pueden provocar fatiga, "
                "dolor de cabeza o disminución en la concentración. "
                "Se sugiere ventilar los espacios de manera constante."
            )

    return {
        "nivel": nivel_css,
        "mensaje": mensaje,
        "mean_pred": round(mean_val, 2),
        "max_pred": round(max_val, 2),
    }


# ======================================================
# TEXTO EXPLICATIVO DE LA PREDICCIÓN
# ======================================================
def generar_texto_explicacion_prediccion(horas=48):
    dias = horas / 24

    return (
        "<strong>Alcance de la predicción futura</strong><br>"
        f"Las estimaciones presentadas corresponden a un horizonte de "
        f"<strong>{int(horas)} horas</strong> "
        f"(aproximadamente <strong>{dias:.1f} días</strong>). "
        "La predicción se construye a partir de los <strong>registros históricos más recientes</strong> "
        "del sensor seleccionado, considerando variables ambientales como "
        "<strong>temperatura, humedad, presión atmosférica y nivel de luz</strong>, "
        "además de patrones temporales (hora del día y día de la semana). "
        "Los valores mostrados representan <strong>escenarios probables</strong> "
        "de la calidad del aire a corto plazo y deben interpretarse como "
        "una <strong>herramienta de apoyo a la toma de decisiones</strong>, "
        "no como valores absolutos garantizados."
    )


# ======================================================
# RESUMEN EJECUTIVO
# ======================================================
def generar_resumen_ejecutivo(
    *,
    total_records,
    pm25_mean,
    pm25_max,
    metrics_pm25=None,
    metrics_co2=None,
):

    # ---------------------------
    # 1. Clasificación sanitaria
    # ---------------------------
    nivel, icono, recomendacion = _clasificar_pm25(pm25_mean)

    # ---------------------------
    # 2. Calidad del modelo PM2.5
    # ---------------------------
    calidad_modelo_pm25 = "no evaluada"

    if metrics_pm25 and "R2" in metrics_pm25:
        r2 = metrics_pm25["R2"]
        if r2 >= 0.8:
            calidad_modelo_pm25 = "alta"
        elif r2 >= 0.6:
            calidad_modelo_pm25 = "adecuada"
        else:
            calidad_modelo_pm25 = "limitada"

    # ---------------------------
    # 3. Calidad del modelo CO2
    # ---------------------------

    calidad_modelo_co2 = None
    if metrics_co2 and "R2" in metrics_co2:
        r2c = metrics_co2["R2"]
        if r2c >= 0.8:
            calidad_modelo_co2 = "alta"
        elif r2c >= 0.6:
            calidad_modelo_co2 = "adecuada"
        else:
            calidad_modelo_co2 = "limitada"

    # ---------------------------
    # 4. Construcción del texto
    # ---------------------------
    sections = [
        (
            "<strong>Confiabilidad del análisis</strong><br>"
            f"Se analizaron <strong>{total_records:,}</strong> registros válidos, "
            "lo que proporciona una base de datos sólida para la evaluación."
        ),
        (
            "<strong>Evaluación de la calidad del aire</strong><br>"
            f"La concentración promedio de PM2.5 fue de "
            f"<strong>{pm25_mean:.2f} µg/m³</strong>, clasificada como "
            f"<strong>{nivel}</strong>."
        ),
        (
            "<strong>Episodios críticos</strong><br>"
            f"El valor máximo registrado fue de "
            f"<strong>{pm25_max:.2f} µg/m³</strong>, lo que indica "
            "momentos puntuales de mayor contaminación."
        ),
        (
            "<strong>Confiabilidad del modelo predictivo (PM2.5)</strong><br>"
            f"El desempeño del modelo se considera <strong>{calidad_modelo_pm25}</strong>, "
            "por lo que las predicciones representan de forma confiable "
            "la tendencia general de los datos."
        ),
    ]

    # --- Sección CO2 solo si existe ---
    if calidad_modelo_co2:
        sections.append(
            (
                "<strong>Confiabilidad del modelo CO₂</strong><br>"
                f"Precisión <strong>{calidad_modelo_co2}</strong>."
            )
        )

    # --- Explicación de la predicción ---
    sections.append(generar_texto_explicacion_prediccion(horas=48))
    # --- Conclusión final ---
    sections.append(("<strong>Conclusión general</strong><br>" f"{recomendacion}"))

    # ---------------------------
    # 5. Fecha de generación
    # ---------------------------
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")

    return {
        "nivel": nivel,
        "icono": icono,
        "mensaje": recomendacion,
        "sections": sections,
        "generated_at": generated_at,
        "footer": f"Reporte generado el {generated_at}",
    }


# ======================================================
# SERVICIO PRINCIPAL
# ======================================================
def ejecutar_analisis(device, start_date=None, end_date=None):
    from core.views import get_mongo_data  # import diferido

    # =========================
    # 1. Query Mongo
    # =========================

    query = {"idDispositivo": device}

    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = (
            pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        )

        query["fecha"] = {"$gte": start_dt, "$lte": end_dt}

    data_list = get_mongo_data(query)

    if not data_list:
        print(">>> ENGINE: NO HAY DATOS")
        return None

    # =========================
    # 2. DataFrame
    # =========================
    df = pd.DataFrame(data_list)

    # -------------------------
    # Normalización columnas
    # -------------------------
    df.rename(
        columns={
            "fecha": "datetime",
            "temperatura": "temp",
            "humedad": "humidity",
            "presion": "pressure",
            "luz": "light",
        },
        inplace=True,
    )

    if "datetime" not in df.columns:
        print("❌ No existe columna datetime")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["temp", "humidity", "pressure", "light", "PM2_5", "CO2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 3. Stats ambientales
    # =========================
    stats = {}
    for col in ["temp", "humidity", "pressure", "light"]:
        if col in df.columns:
            stats[col] = round(df[col].mean(), 2)

    # =========================
    # 4. Modelos (SIN recortar datos)
    # =========================

    m25, p25, model_pm25 = entrenar_modelo(df.copy(), "PM2_5")
    mco, pco, model_co2 = entrenar_modelo(df.copy(), "CO2")

    # ==========================
    # 4.1 PREDICCIÓN FUTURA
    # ==========================

    future_pm25 = None
    future_co2 = None

    # --- Intento con modelo ML ---
    try:
        if model_pm25:
            fut_features = generar_features_futuras(df, steps=48)
            if fut_features is not None:
                future_pm25 = predecir_futuro(model_pm25, fut_features)

        if model_co2:
            fut_features = generar_features_futuras(df, steps=48)
            if fut_features is not None:
                future_co2 = predecir_futuro(model_co2, fut_features)
    except Exception as e:
        print("⚠️ Error en predicción futura ML:", e)

    # --- Fallback SIEMPRE ---
    if future_pm25 is None:
        future_pm25 = predecir_futuro_simple(df, "PM2_5", steps=48)

    if future_co2 is None:
        future_co2 = predecir_futuro_simple(df, "CO2", steps=48)

    # ==========================
    # 4.2 TEXTOS PARA DASHBOARD
    # ==========================

    txt_pm25 = generar_texto_dashboard(future_pm25, "PM2_5")
    txt_co2 = generar_texto_dashboard(future_co2, "CO2")

    # ==========================
    # 5. RESUMEN EJECUTIVO
    # ==========================

    resumen = None
    if p25 is not None and not p25.empty and "y_true" in p25.columns:
        pm25_mean = p25["y_true"].mean()
        pm25_max = p25["y_true"].max()

        resumen = generar_resumen_ejecutivo(
            total_records=len(p25),
            pm25_mean=pm25_mean,
            pm25_max=pm25_max,
            metrics_pm25=m25,
            metrics_co2=mco,
        )

    return {
        "stats_ambientales": stats,
        "metrics_pm25": m25,
        "preds_pm25": p25,
        "metrics_co2": mco,
        "preds_co2": pco,
        "future_pm25": future_pm25,
        "future_co2": future_co2,
        "txt_pm25": txt_pm25,
        "txt_co2": txt_co2,
        "resumen_ejecutivo": resumen,
    }


# ======================================================
# COMMAND
# ======================================================
class Command(BaseCommand):
    help = "Motor de análisis predictivo (engine ML)"

    def add_arguments(self, parser):
        parser.add_argument("--device", type=str, required=True)
        parser.add_argument("--start", type=str)
        parser.add_argument("--end", type=str)

    def handle(self, *args, **options):
        device = options["device"]
        start = options.get("start")
        end = options.get("end")

        self.stdout.write(self.style.SUCCESS("🚀 Ejecutando motor ML"))

        result = ejecutar_analisis(device, start, end)

        if not result:
            self.stdout.write(self.style.WARNING("⚠️ No se generaron resultados"))
            return

        self.stdout.write(self.style.SUCCESS("✅ Proceso completado"))
