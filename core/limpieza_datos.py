"""
core/limpieza_datos.py
Módulo centralizado de limpieza y validación de datos para Machine Learning y análisis.
Fase A: Validación de rangos físicos, remuestreo horario, selección por cobertura real.
"""

import math
import numpy as np
import pandas as pd

# ==============================================================================
# 1. PARÁMETROS Y RANGOS FÍSICOS CONFIRMADOS (P1 - P4)
# ==============================================================================

RANGOS_FISICOS = {
    "PM2_5": (0.0, 1000.0),       # 0.0 es válido (aire limpio)
    "PM10": (0.0, 1000.0),        # 0.0 es válido
    "CO2": (350.0, 5000.0),       # 0.0 o <350 es falla de sensor
    "temp": (-5.0, 45.0),
    "humidity": (0.0, 100.0),
    "pressure": (800.0, 1016.0),  # Presión barométrica absoluta de estación
    "light": (0.0, 150000.0),     # 0 es válido (noche o sensor tapado)
}

COBERTURA_MINIMA = 0.60           # 60% de horas con dato en el periodo
MIN_HORAS_ENTRENAMIENTO = 50      # Umbral mínimo de horas para entrenar RandomForest

COLUMNAS_AMBIENTALES = ["temp", "humidity", "pressure", "light"]
TARGETS_PREDICCION = ["PM2_5", "CO2"]

RENAME_MAP = {
    "fecha": "datetime",
    "date": "datetime",
    "timestamp": "datetime",
    "temperatura": "temp",
    "temperature": "temp",
    "temp": "temp",
    "humedad": "humidity",
    "humidity": "humidity",
    "hum": "humidity",
    "presion": "pressure",
    "pressure": "pressure",
    "luz": "light",
    "light": "light",
    "illumination": "light",
    "co2": "CO2",
    "eco2": "CO2",
    "CO2": "CO2",
    "pm2_5": "PM2_5",
    "pm2.5": "PM2_5",
    "pm25": "PM2_5",
    "PM2_5": "PM2_5",
    "pm10": "PM10",
    "PM10": "PM10",
    "iddispositivo": "idDispositivo",
    "deviceid": "idDispositivo",
    "idDispositivo": "idDispositivo",
}


class ResultadoLimpieza:
    """Contenedor de resultados del pipeline de limpieza."""
    def __init__(self, df_horario, features_validas, targets_validos, reporte):
        self.df = df_horario
        self.features_validas = features_validas
        self.targets_validos = targets_validos
        self.reporte = reporte


def limpiar_lecturas(df_crudo, targets=None):
    """
    Pipeline de limpieza puro, desacoplado y sin imputación destructiva.

    Pasos:
      1. Normalización de nombres de columnas.
      2. Conversión de fecha y tipado numérico (errors='coerce').
      3. Deduplicación exacta por (idDispositivo, datetime).
      4. Filtrado por rangos físicos de validez (valores fuera de rango -> NaN).
      5. Remuestreo horario por promedio.
      6. Evaluación de cobertura mínima (>= 60%).
      7. Selección dinámica de features y validación de targets.

    Retorna:
      ResultadoLimpieza con:
        - df: DataFrame horario limpio.
        - features_validas: lista de variables ambientales con >= 60% cobertura.
        - targets_validos: lista de targets con >= 60% cobertura.
        - reporte: diccionario detallado con auditoría de la limpieza y motivos.
    """
    if targets is None:
        targets = TARGETS_PREDICCION.copy()

    reporte = {
        "filas_recibidas": 0,
        "filas_fecha_invalida": 0,
        "filas_duplicadas": 0,
        "horas_totales_periodo": 0,
        "horas_con_datos": 0,
        "anulados_por_rango": {},
        "cobertura_por_columna": {},
        "features_descartadas": {},
        "targets_descartados": {},
        "motivo_fallback": {},
    }

    if df_crudo is None or df_crudo.empty:
        for t in targets:
            reporte["motivo_fallback"][t] = "No se recibieron datos de entrada (DataFrame vacío)."
        return ResultadoLimpieza(pd.DataFrame(), [], [], reporte)

    df = df_crudo.copy()
    reporte["filas_recibidas"] = len(df)

    # 1. Normalización de nombres
    df.columns = df.columns.str.lower().str.strip()
    nuevos_nombres = {k.lower(): v for k, v in RENAME_MAP.items() if k.lower() in df.columns}
    df.rename(columns=nuevos_nombres, inplace=True)

    if "datetime" not in df.columns:
        for t in targets:
            reporte["motivo_fallback"][t] = "Columna de fecha/datetime ausente en los datos."
        return ResultadoLimpieza(pd.DataFrame(), [], [], reporte)

    # 2. Tipado de fecha y filtrado de inválidos
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    nulos_fecha = int(df["datetime"].isna().sum())
    reporte["filas_fecha_invalida"] = nulos_fecha
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    if df.empty:
        for t in targets:
            reporte["motivo_fallback"][t] = "Todas las filas tienen fecha inválida."
        return ResultadoLimpieza(pd.DataFrame(), [], [], reporte)

    # 3. Deduplicación exacta
    cols_dedup = ["idDispositivo", "datetime"] if "idDispositivo" in df.columns else ["datetime"]
    antes_dedup = len(df)
    df = df.drop_duplicates(subset=cols_dedup)
    reporte["filas_duplicadas"] = antes_dedup - len(df)

    # 4. Tipado numérico y filtro por rangos físicos
    cols_a_validar = list(RANGOS_FISICOS.keys())
    for col in cols_a_validar:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            min_val, max_val = RANGOS_FISICOS[col]
            
            # Máscara de valores fuera de rango físico
            # Para CO2, <= 350 es considerado fuera de rango o falla
            if col == "CO2":
                mask_invalido = (df[col] < min_val) | (df[col] > max_val)
            else:
                mask_invalido = (df[col] < min_val) | (df[col] > max_val)

            conteo_invalidos = int(mask_invalido.sum())
            reporte["anulados_por_rango"][col] = conteo_invalidos
            df.loc[mask_invalido, col] = np.nan
        else:
            reporte["anulados_por_rango"][col] = 0

    # 5. Remuestreo horario por promedio
    # Usamos sólo columnas numéricas presentes
    cols_numericas_presentes = [c for c in cols_a_validar if c in df.columns]
    if not cols_numericas_presentes:
        for t in targets:
            reporte["motivo_fallback"][t] = "No existen columnas numéricas de interés en el conjunto de datos."
        return ResultadoLimpieza(pd.DataFrame(), [], [], reporte)

    # Establecemos índice temporal para resample
    df_horario = df.set_index("datetime")[cols_numericas_presentes].resample("1h").mean()
    reporte["horas_totales_periodo"] = len(df_horario)

    # Eliminar horas que quedaron totalmente vacías
    df_horario_con_datos = df_horario.dropna(how="all")
    reporte["horas_con_datos"] = len(df_horario_con_datos)

    if df_horario.empty or len(df_horario) == 0:
        for t in targets:
            reporte["motivo_fallback"][t] = "Periodo sin horas válidas tras el remuestreo."
        return ResultadoLimpieza(pd.DataFrame(), [], [], reporte)

    # 6. Cálculo de cobertura y selección por umbral (>= 60%)
    horas_totales = len(df_horario)
    features_validas = []
    targets_validos = []

    for col in COLUMNAS_AMBIENTALES:
        if col in df_horario.columns:
            horas_validas_col = int(df_horario[col].notna().sum())
            cobertura = horas_validas_col / horas_totales
            reporte["cobertura_por_columna"][col] = round(cobertura, 4)

            if cobertura >= COBERTURA_MINIMA:
                features_validas.append(col)
            else:
                reporte["features_descartadas"][col] = (
                    f"Cobertura insuficiente ({cobertura * 100:.1f}% < {COBERTURA_MINIMA * 100:.0f}%)."
                )
        else:
            reporte["cobertura_por_columna"][col] = 0.0
            reporte["features_descartadas"][col] = "Columna no presente en los datos."

    for t in targets:
        if t in df_horario.columns:
            horas_validas_t = int(df_horario[t].notna().sum())
            cobertura_t = horas_validas_t / horas_totales
            reporte["cobertura_por_columna"][t] = round(cobertura_t, 4)

            if cobertura_t >= COBERTURA_MINIMA:
                targets_validos.append(t)
            else:
                reporte["targets_descartados"][t] = (
                    f"Cobertura insuficiente ({cobertura_t * 100:.1f}% < {COBERTURA_MINIMA * 100:.0f}%)."
                )
                reporte["motivo_fallback"][t] = (
                    f"Target '{t}' no alcanza la cobertura mínima de {COBERTURA_MINIMA * 100:.0f}% "
                    f"(obtuvo {cobertura_t * 100:.1f}%)."
                )
        else:
            reporte["cobertura_por_columna"][t] = 0.0
            reporte["targets_descartados"][t] = "Columna target no presente en los datos."
            reporte["motivo_fallback"][t] = f"Target '{t}' no existe en las mediciones del sensor."

    # Validar si no hay variables ambientales para entrenar
    if not features_validas:
        for t in targets_validos:
            reporte["motivo_fallback"][t] = (
                "Ninguna variable ambiental (temp, humidity, pressure, light) alcanzó la cobertura mínima del 60%."
            )

    # Devolvemos el dataframe horario con la fecha de vuelta como columna 'datetime'
    df_resultado = df_horario.reset_index()

    return ResultadoLimpieza(
        df_horario=df_resultado,
        features_validas=features_validas,
        targets_validos=targets_validos,
        reporte=reporte
    )

