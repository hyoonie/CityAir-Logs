import math
from datetime import datetime

import pandas as pd
from django.core.management.base import BaseCommand
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.limpieza_datos import (
    limpiar_lecturas,
    MIN_HORAS_ENTRENAMIENTO,
    COLUMNAS_AMBIENTALES,
)


# ======================================================
# FEATURES DISPONIBLES
# ======================================================
def get_available_features(df):
    return [c for c in COLUMNAS_AMBIENTALES if c in df.columns]


# ======================================================
# ENTRENAMIENTO
# ======================================================
def entrenar_modelo(df, target, features_validas=None):
    """
    Entrena modelo en memoria de forma segura con variables validadas por cobertura.
    Crea variables temporales antes de limpiar con dropna.
    """

    if "datetime" not in df.columns:
        return None, None, None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Variables temporales
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek

    # Si se especifican features validadas por cobertura, se usan esas; si no, fallback a las que existan en df
    if features_validas is not None:
        env_features = [f for f in features_validas if f in df.columns]
    else:
        env_features = [f for f in COLUMNAS_AMBIENTALES if f in df.columns]

    time_features = ["hour", "dow"]
    features = env_features + time_features

    if not env_features:
        print(f"⚠️ No hay features ambientales válidas para entrenar {target}")
        return None, None, None

    # -------------------------
    # 3. Limpieza segura
    # -------------------------

    required_cols = ["datetime", target] + features

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas para {target}: {missing}")
        return None, None, None

    df_clean = df.dropna(subset=required_cols)

    if len(df_clean) < MIN_HORAS_ENTRENAMIENTO:
        print(f"⚠️ Horas insuficientes para {target}: {len(df_clean)} < {MIN_HORAS_ENTRENAMIENTO}")
        return None, None, None

    # -------------------------
    # 4. Preparación X, y
    # -------------------------

    X = df_clean[features]
    y = df_clean[target]

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
        "n_samples": len(df_clean),
    }

    # -------------------------
    # 7. Predicciones
    # -------------------------

    preds = pd.DataFrame(
        {
            "datetime": df_clean["datetime"],
            "y_true": y,
            "y_pred": yhat,
        }
    ).sort_values("datetime")

    return metrics, preds, rf


# ======================================================
# GENERACIÓN DE FEATURES FUTURAS (ROBUSTA)
# ======================================================
def generar_features_futuras(df, features_validas=None, steps=48):
    """
    Genera features futuras de forma segura para predicción.
    Usa persistencia + variables temporales basándose únicamente en las features válidas.
    """

    if df is None or df.empty:
        print("⚠️ generar_features_futuras: DF vacío")
        return None

    if "datetime" not in df.columns:
        print("⚠️ generar_features_futuras: no existe datetime")
        return None

    df = df.sort_values("datetime")

    last_row = df.iloc[-1]

    # Validar que exista al menos una variable ambiental válida
    if features_validas is not None:
        env_cols = [c for c in features_validas if c in df.columns]
    else:
        env_cols = [c for c in COLUMNAS_AMBIENTALES if c in df.columns]

    if not env_cols:
        print("⚠️ generar_features_futuras: no hay columnas ambientales válidas")
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
    con interpretación práctica estructurada para el usuario.
    """
    if future_df is None or future_df.empty:
        return None

    mean_val = future_df["y_pred"].mean()
    max_val = future_df["y_pred"].max()

    # ======================================================
    # PM2.5
    # ======================================================
    if contaminante == "PM2_5":

        # Clasificación sanitaria (referencia OMS aproximada)
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

        # -------- Texto estructurado --------
        mensaje = (
            "<div class='pred-card'>"
            # --- Descripción general ---
            "<div class='pred-section'>"
            "<div class='pred-title'>📌 Descripción general</div>"
            "Esta proyección estima el comportamiento de la concentración de "
            "<strong>PM2.5</strong> durante las próximas <strong>48 horas</strong>, "
            "a partir de los registros históricos recientes y las condiciones "
            "ambientales observadas."
            "</div>"
            # --- Resultado del modelo ---
            "<div class='pred-section'>"
            "<div class='pred-title'>📊 Resultado del modelo</div>"
            "<ul class='pred-bullets'>"
            f"<li><strong>Nivel estimado:</strong> {nivel_txt.capitalize()}</li>"
            f"<li><strong>Promedio esperado:</strong> {mean_val:.1f} µg/m³</li>"
            f"<li><strong>Pico máximo esperado:</strong> {max_val:.1f} µg/m³</li>"
            "</ul>"
            "</div>"
        )

        # -------- Interpretación práctica --------
        if nivel_txt == "bueno":
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🩺 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>La calidad del aire es <strong>adecuada</strong> para la población general.</li>"
                "<li>No se esperan efectos adversos incluso durante actividades "
                "al aire libre de intensidad moderada.</li>"
                "<li>Escenario favorable para actividades escolares, recreativas "
                "y laborales en exteriores.</li>"
                "</ul>"
                "</div>"
            )

        elif nivel_txt == "moderado":
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🩺 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>La calidad del aire es <strong>aceptable</strong> para la mayoría de la población.</li>"
                "<li>Personas con <strong>asma, alergias o sensibilidad respiratoria</strong> "
                "podrían presentar molestias leves.</li>"
                "<li>Se recomienda <strong>evitar ejercicio intenso</strong> "
                "en exteriores en horas de mayor actividad urbana.</li>"
                "<li>Priorizar <strong>espacios bien ventilados</strong> como medida preventiva.</li>"
                "</ul>"
                "</div>"
            )

        elif nivel_txt == "alto":
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🩺 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>La calidad del aire es <strong>poco saludable</strong> "
                "para grupos vulnerables.</li>"
                "<li>Niños, adultos mayores y personas con enfermedades respiratorias "
                "o cardiovasculares pueden presentar mayor riesgo.</li>"
                "<li>Se aconseja <strong>limitar actividades físicas</strong> al aire libre.</li>"
                "<li>Preferir permanecer en <strong>espacios interiores</strong> "
                "durante periodos de mayor concentración.</li>"
                "</ul>"
                "</div>"
            )

        else:  # muy alto
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🩺 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>Este escenario representa un <strong>riesgo potencial</strong> "
                "para la salud general.</li>"
                "<li>La exposición prolongada puede provocar síntomas respiratorios "
                "incluso en personas sanas.</li>"
                "<li>Se recomienda <strong>evitar actividades en exteriores</strong>.</li>"
                "<li>Mantener puertas y ventanas cerradas en zonas de alta contaminación.</li>"
                "<li>Seguir indicaciones de <strong>autoridades sanitarias</strong>.</li>"
                "</ul>"
                "</div>"
            )

        mensaje += "</div>"  # cierre pred-card

    # ======================================================
    # CO2
    # ======================================================
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

        # -------- Texto estructurado --------
        mensaje = (
            "<div class='pred-card'>"
            # --- Descripción general ---
            "<div class='pred-section'>"
            "<div class='pred-title'>📌 Descripción general</div>"
            "Esta proyección estima la evolución de los niveles de "
            "<strong>CO₂</strong> durante las próximas <strong>48 horas</strong>, "
            "a partir de los datos recientes del sensor."
            "</div>"
            # --- Resultado del modelo ---
            "<div class='pred-section'>"
            "<div class='pred-title'>📊 Resultado del modelo</div>"
            "<ul class='pred-bullets'>"
            f"<li><strong>Nivel estimado:</strong> {nivel_txt.capitalize()}</li>"
            f"<li><strong>Promedio esperado:</strong> {mean_val:.0f} ppm</li>"
            f"<li><strong>Pico máximo esperado:</strong> {max_val:.0f} ppm</li>"
            "</ul>"
            "</div>"
        )

        # -------- Interpretación práctica --------
        if nivel_txt == "normal":
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🏢 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>Las concentraciones son <strong>adecuadas</strong> para espacios "
                "interiores y exteriores.</li>"
                "<li>No se asocian con efectos negativos en el bienestar.</li>"
                "<li>Permiten mantener un buen nivel de <strong>confort ambiental</strong>.</li>"
                "</ul>"
                "</div>"
            )

        elif nivel_txt == "moderado":
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🏢 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>No representan un riesgo inmediato para la salud.</li>"
                "<li>Pueden generar sensación de <strong>aire cargado</strong> "
                "y ligera disminución en la concentración.</li>"
                "<li>Se recomienda <strong>mejorar la ventilación</strong> "
                "en aulas, oficinas y espacios cerrados.</li>"
                "</ul>"
                "</div>"
            )

        else:  # elevado
            mensaje += (
                "<div class='pred-section'>"
                "<div class='pred-title'>🏢 Interpretación práctica</div>"
                "<ul class='pred-bullets'>"
                "<li>Concentraciones elevadas pueden provocar "
                "<strong>fatiga y dolor de cabeza</strong>.</li>"
                "<li>También se asocian con reducción en la capacidad de atención.</li>"
                "<li>Se aconseja <strong>ventilar inmediatamente</strong> "
                "los espacios cerrados.</li>"
                "<li>Reducir la ocupación cuando sea posible.</li>"
                "</ul>"
                "</div>"
            )

        mensaje += "</div>"  # cierre pred-card

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


def generar_resumen_ejecutivo(
    *,
    total_records,
    pm25_mean,
    pm25_max,
    metrics_pm25=None,
    metrics_co2=None,
):

    nivel, icono, recomendacion = _clasificar_pm25(pm25_mean)

    # ---------------------------
    # Calidad modelo PM2.5
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
    # Calidad modelo CO2
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

    sections = []

    # ---------------------------
    # Bloque 1 — Confiabilidad del análisis
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>📊 Confiabilidad del análisis</div>"
        "<ul class='res-bullets'>"
        f"<li>Se analizaron <strong>{total_records:,}</strong> registros válidos obtenidos de los sensores.</li>"
        "<li>Este volumen de información permite realizar un análisis con un "
        "<strong>alto nivel de confiabilidad estadística</strong>.</li>"
        "<li>Los resultados presentados reflejan tendencias generales del comportamiento ambiental.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    # ---------------------------
    # Bloque 2 — Evaluación sanitaria
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>🌫 Evaluación de la calidad del aire</div>"
        "<ul class='res-bullets'>"
        f"<li>La concentración promedio de partículas PM2.5 fue de "
        f"<strong>{pm25_mean:.2f} µg/m³</strong>.</li>"
        f"<li>De acuerdo con estándares de referencia, esta condición se clasifica como "
        f"<strong>{nivel}</strong>.</li>"
        "<li>Este indicador permite estimar el impacto potencial en la salud de la población.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    # ---------------------------
    # Bloque 3 — Episodios críticos
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>⚠️ Episodios críticos detectados</div>"
        "<ul class='res-bullets'>"
        f"<li>Se registró un valor máximo de PM2.5 de "
        f"<strong>{pm25_max:.2f} µg/m³</strong>.</li>"
        "<li>Estos picos representan <strong>eventos aislados de mayor contaminación</strong>.</li>"
        "<li>Aunque no son permanentes, pueden implicar "
        "<strong>riesgos temporales para la salud</strong>, especialmente en personas sensibles.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    # ---------------------------
    # Bloque 4 — Desempeño del modelo PM2.5
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>🤖 Desempeño del modelo predictivo (PM2.5)</div>"
        "<ul class='res-bullets'>"
        f"<li>La calidad del modelo se considera <strong>{calidad_modelo_pm25}</strong>.</li>"
        "<li>Esto indica que las predicciones reproducen de manera "
        "<strong>adecuada la tendencia general</strong> de los datos observados.</li>"
        "<li>El modelo es una herramienta de apoyo para la toma de decisiones preventivas.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    # ---------------------------
    # Bloque 5 — Desempeño del modelo CO2
    # ---------------------------
    if calidad_modelo_co2:
        sections.append(
            "<div class='res-card'>"
            "<div class='res-section'>"
            "<div class='res-title'>🏢 Desempeño del modelo predictivo (CO₂)</div>"
            "<ul class='res-bullets'>"
            f"<li>La precisión estimada del modelo es <strong>{calidad_modelo_co2}</strong>.</li>"
            "<li>Este análisis resulta especialmente útil para evaluar "
            "<strong>condiciones de confort ambiental</strong> en espacios interiores.</li>"
            "<li>Contribuye a la gestión de ventilación y calidad del aire en interiores.</li>"
            "</ul>"
            "</div>"
            "</div>"
        )

    # ---------------------------
    # Bloque 6 — Alcance de la predicción
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>⏱ Alcance de la predicción futura</div>"
        "<ul class='res-bullets'>"
        "<li>Las estimaciones cubren un horizonte aproximado de "
        "<strong>48 horas</strong> (2 días).</li>"
        "<li>Se basan en el comportamiento histórico más reciente de las variables ambientales.</li>"
        "<li>Incluyen factores como "
        "<strong>temperatura, humedad, presión atmosférica y nivel de iluminación</strong>.</li>"
        "<li>También consideran patrones temporales como "
        "<strong>hora del día</strong> y <strong>día de la semana</strong>.</li>"
        "<li>Las predicciones representan <strong>escenarios probables</strong> y no valores absolutos.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

    # ---------------------------
    # Bloque 7 — Conclusión general
    # ---------------------------
    sections.append(
        "<div class='res-card'>"
        "<div class='res-section'>"
        "<div class='res-title'>🧾 Conclusión general</div>"
        "<ul class='res-bullets'>"
        f"<li>{recomendacion}</li>"
        "<li>Se recomienda utilizar esta información como apoyo para "
        "<strong>acciones preventivas y de concientización ambiental</strong>.</li>"
        "</ul>"
        "</div>"
        "</div>"
    )

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
    # 2. DataFrame y Limpieza Centralizada (Fase A)
    # =========================
    df_raw = pd.DataFrame(data_list)

    resultado_limpieza = limpiar_lecturas(df_raw, targets=["PM2_5", "CO2"])
    df = resultado_limpieza.df
    features_validas = resultado_limpieza.features_validas
    targets_validos = resultado_limpieza.targets_validos
    reporte_limpieza = resultado_limpieza.reporte

    if df.empty or "datetime" not in df.columns:
        print("❌ Dataset limpio vacío o sin columna datetime")
        return None

    # =========================
    # 3. Stats ambientales
    # =========================
    stats = {}
    for col in COLUMNAS_AMBIENTALES:
        if col in df.columns and pd.notna(df[col]).any():
            stats[col] = round(float(df[col].mean()), 2)

    # =========================
    # 4. Modelos (Solo si el target tiene cobertura >= 60%)
    # =========================
    m25, p25, model_pm25 = None, None, None
    mco, pco, model_co2 = None, None, None

    if "PM2_5" in targets_validos:
        m25, p25, model_pm25 = entrenar_modelo(df.copy(), "PM2_5", features_validas=features_validas)
        if m25 is None:
            reporte_limpieza["motivo_fallback"]["PM2_5"] = (
                f"Horas con datos completos insuficientes (< {MIN_HORAS_ENTRENAMIENTO} h)."
            )
    else:
        print(f"⚠️ PM2_5 no cumple cobertura mínima: {reporte_limpieza['motivo_fallback'].get('PM2_5')}")

    if "CO2" in targets_validos:
        mco, pco, model_co2 = entrenar_modelo(df.copy(), "CO2", features_validas=features_validas)
        if mco is None:
            reporte_limpieza["motivo_fallback"]["CO2"] = (
                f"Horas con datos completos insuficientes (< {MIN_HORAS_ENTRENAMIENTO} h)."
            )
    else:
        print(f"⚠️ CO2 no cumple cobertura mínima: {reporte_limpieza['motivo_fallback'].get('CO2')}")

    # ==========================
    # 4.1 PREDICCIÓN FUTURA
    # ==========================

    future_pm25 = None
    future_co2 = None

    # --- Intento con modelo ML ---
    try:
        if model_pm25:
            fut_features_pm25 = generar_features_futuras(df, features_validas=features_validas, steps=48)
            if fut_features_pm25 is not None:
                future_pm25 = predecir_futuro(model_pm25, fut_features_pm25)

        if model_co2:
            fut_features_co2 = generar_features_futuras(df, features_validas=features_validas, steps=48)
            if fut_features_co2 is not None:
                future_co2 = predecir_futuro(model_co2, fut_features_co2)
    except Exception as e:
        print("⚠️ Error en predicción futura ML:", e)

    # --- Fallback SIEMPRE si no se generó con ML ---
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
        "reporte_limpieza": reporte_limpieza,
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
