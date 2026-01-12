import pandas as pd

from core.ml.engine_live import entrenar_modelo


def ejecutar_analisis(device, start_date=None, end_date=None):
    # 👇 IMPORT DIFERIDO (rompe el ciclo)
    from core.views import get_mongo_data

    print(">>> ML SERVICE: INICIO")

    # =========================
    # 1. Query Mongo
    # =========================
    query = {"idDispositivo": device}

    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        # 🔥 FIX CLAVE: incluir TODO el día final
        end_dt = (
            pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        )

        query["fecha"] = {
            "$gte": start_dt,
            "$lte": end_dt,
        }

        print(">>> ML SERVICE: RANGO REAL:", start_dt, "→", end_dt)

    print(">>> ML SERVICE: EJECUTANDO QUERY:", query)
    data_list = get_mongo_data(query)

    print(">>> ML SERVICE: QUERY OK")
    print(">>> ML SERVICE: DATOS ENCONTRADOS:", len(data_list) if data_list else 0)

    if not data_list:
        print(">>> ML SERVICE: NO HAY DATOS")
        return None

    # =========================
    # 2. DataFrame
    # =========================
    df = pd.DataFrame(data_list)

    print(">>> DF COLUMNS:", df.columns.tolist())
    print(">>> DF LEN:", len(df))

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

    print(">>> ML SERVICE: STATS:", stats)

    # =========================
    # 4. Modelos (SIN recortar datos)
    # =========================
    print(">>> ML SERVICE: ENTRENANDO PM2_5")
    m25, p25 = entrenar_modelo(df.copy(), "PM2_5")

    print(">>> ML SERVICE: ENTRENANDO CO2")
    mco, pco = entrenar_modelo(df.copy(), "CO2")

    print(">>> ML SERVICE: FIN OK")

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
        print(">>> RESUMEN GENERADO:", resumen)

        return {
            "stats_ambientales": stats,
            "metrics_pm25": m25,
            "preds_pm25": p25,
            "metrics_co2": mco,
            "preds_co2": pco,
            "resumen_ejecutivo": resumen,
        }


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


from datetime import datetime


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
                "<strong>Confiabilidad del modelo predictivo (CO₂)</strong><br>"
                f"El modelo de CO₂ mostró una precisión <strong>{calidad_modelo_co2}</strong>, "
                "aportando una referencia adicional sobre la calidad del ambiente."
            )
        )

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
