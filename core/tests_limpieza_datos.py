"""
core/tests_limpieza_datos.py
Pruebas unitarias para el módulo centralizado de limpieza de datos (Fase A)
y su integración con el motor predictivo (RandomForest).
"""

import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from unittest.mock import patch

from core.limpieza_datos import (
    limpiar_lecturas,
    RANGOS_FISICOS,
    COBERTURA_MINIMA,
    MIN_HORAS_ENTRENAMIENTO,
)
from core.management.commands.run_engine import (
    entrenar_modelo,
    ejecutar_analisis,
    generar_features_futuras,
)


class LimpiezaDatosUnitTest(SimpleTestCase):
    """Pruebas directas de las funciones y reglas de core/limpieza_datos.py."""

    def _generar_dataset_sintetico(self, num_horas=60, freq="1h", id_dispositivo="sensor_test"):
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=num_horas, freq=freq)
        data = {
            "idDispositivo": id_dispositivo,
            "fecha": fechas,
            "temperatura": np.random.uniform(15.0, 25.0, size=num_horas),
            "humedad": np.random.uniform(50.0, 80.0, size=num_horas),
            "presion": np.random.uniform(840.0, 870.0, size=num_horas),
            "luz": np.random.uniform(0.0, 50000.0, size=num_horas),
            "PM2_5": np.random.uniform(5.0, 45.0, size=num_horas),
            "CO2": np.random.uniform(400.0, 900.0, size=num_horas),
        }
        return pd.DataFrame(data)

    def test_h1_dispositivo_api_presion_y_luz_vacias(self):
        """
        Regresión H1: Un dispositivo tipo API REST registra 'presion' y 'luz' como None.
        El pipeline debe conservar las filas y seleccionar sólo temp y humidity como features.
        """
        df = self._generar_dataset_sintetico(num_horas=60)
        df["presion"] = None
        df["luz"] = None

        res = limpiar_lecturas(df)

        # Debe conservar las 60 horas
        self.assertEqual(len(res.df), 60)
        # Features válidas deben incluir temp y humidity, pero NO pressure ni light
        self.assertIn("temp", res.features_validas)
        self.assertIn("humidity", res.features_validas)
        self.assertNotIn("pressure", res.features_validas)
        self.assertNotIn("light", res.features_validas)
        # Targets válidos presentes
        self.assertIn("PM2_5", res.targets_validos)
        self.assertIn("CO2", res.targets_validos)

    def test_rangos_fisicos_anulan_valores_imposibles(self):
        """
        Verifica que los valores fuera de rango pasen a NaN (sin borrar la fila completa)
        y que los ceros válidos (PM2_5=0, light=0) y picos plausibles se conserven.
        """
        df = self._generar_dataset_sintetico(num_horas=10)
        # Valores extremos o imposibles
        df.loc[0, "temperatura"] = -50.0   # Inválido (< -5)
        df.loc[1, "temperatura"] = 70.0    # Inválido (> 45)
        df.loc[2, "humedad"] = 140.0       # Inválido (> 100)
        df.loc[3, "presion"] = 500.0       # Inválido (< 800)
        df.loc[4, "CO2"] = 0.0             # Inválido (CO2 nunca es 0 en aire)
        df.loc[5, "CO2"] = 200.0           # Inválido (< 350)
        # Valores válidos que no deben anularse
        df.loc[6, "PM2_5"] = 0.0           # Válido (aire limpio)
        df.loc[7, "PM2_5"] = 350.0         # Válido (pico plausible / pirotecnia)
        df.loc[8, "luz"] = 0.0             # Válido (noche)

        res = limpiar_lecturas(df)

        reporte = res.reporte
        self.assertGreaterEqual(reporte["anulados_por_rango"]["temp"], 2)
        self.assertGreaterEqual(reporte["anulados_por_rango"]["humidity"], 1)
        self.assertGreaterEqual(reporte["anulados_por_rango"]["pressure"], 1)
        self.assertGreaterEqual(reporte["anulados_por_rango"]["CO2"], 2)

        # Comprobar que en el DF resultante hora 6 tiene PM2_5 = 0.0 y hora 8 tiene light = 0.0
        row_6 = res.df.iloc[6]
        self.assertEqual(row_6["PM2_5"], 0.0)
        row_8 = res.df.iloc[8]
        self.assertEqual(row_8["light"], 0.0)

    def test_cobertura_minima_60_porciento(self):
        """
        Si una variable tiene menos del 60% de cobertura horaria, se descarta.
        Si un target tiene menos del 60%, se descarta y se registra motivo_fallback.
        """
        df = self._generar_dataset_sintetico(num_horas=100)
        # Dejar 'pressure' con solo 30 horas válidas (30% cobertura)
        df.loc[30:, "presion"] = np.nan
        # Dejar 'CO2' con solo 40 horas válidas (40% cobertura)
        df.loc[40:, "CO2"] = np.nan

        res = limpiar_lecturas(df)

        self.assertNotIn("pressure", res.features_validas)
        self.assertIn("pressure", res.reporte["features_descartadas"])
        self.assertNotIn("CO2", res.targets_validos)
        self.assertIn("CO2", res.reporte["motivo_fallback"])
        self.assertIn("cobertura", res.reporte["motivo_fallback"]["CO2"].lower())

    def test_deduplicacion_exacta(self):
        """Lecturas duplicadas con la misma fecha y dispositivo se eliminan."""
        df = self._generar_dataset_sintetico(num_horas=20)
        duplicados = df.iloc[:5].copy()
        df_con_dups = pd.concat([df, duplicados], ignore_index=True)

        res = limpiar_lecturas(df_con_dups)
        self.assertEqual(res.reporte["filas_duplicadas"], 5)

    def test_remuestreo_horario(self):
        """Lecturas cada 10 minutos (6 por hora) se agrupan en promedios horarios."""
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=60, freq="10min")  # 10 horas
        df = pd.DataFrame({
            "datetime": fechas,
            "temp": 20.0,
            "humidity": 60.0,
            "PM2_5": 10.0,
            "CO2": 450.0,
        })
        res = limpiar_lecturas(df)
        # 60 lecturas de 10 min corresponden a 10 horas
        self.assertEqual(len(res.df), 10)
        self.assertEqual(res.df["temp"].iloc[0], 20.0)

    def test_dataframe_vacio_o_sin_fecha(self):
        """Casos degenerados no deben generar excepciones no controladas."""
        res_vacio = limpiar_lecturas(pd.DataFrame())
        self.assertTrue(res_vacio.df.empty)
        self.assertIn("PM2_5", res_vacio.reporte["motivo_fallback"])

        df_sin_fecha = pd.DataFrame({"temp": [20, 21], "hum": [50, 55]})
        res_sin_fecha = limpiar_lecturas(df_sin_fecha)
        self.assertTrue(res_sin_fecha.df.empty)
        self.assertIn("PM2_5", res_sin_fecha.reporte["motivo_fallback"])


class RunEngineIntegrationTest(SimpleTestCase):
    """Pruebas de integración del pipeline con entrenar_modelo y ejecutar_analisis."""

    def test_entrenar_modelo_con_features_dinamicas_api(self):
        """
        Valida que entrenar_modelo funcione con éxito en un sensor tipo API
        que solo cuenta con 'temp' y 'humidity'.
        """
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=60, freq="1h")
        df_limpio = pd.DataFrame({
            "datetime": fechas,
            "temp": np.random.uniform(15, 25, 60),
            "humidity": np.random.uniform(40, 70, 60),
            "PM2_5": np.random.uniform(5, 30, 60),
        })

        features_validas = ["temp", "humidity"]
        metrics, preds, model = entrenar_modelo(df_limpio, "PM2_5", features_validas=features_validas)

        self.assertIsNotNone(metrics)
        self.assertIsNotNone(preds)
        self.assertIsNotNone(model)
        self.assertIn("R2", metrics)
        self.assertEqual(metrics["n_samples"], 60)
        # Las features entrenadas deben ser temp, humidity, hour, dow
        self.assertEqual(metrics["features"], ["temp", "humidity", "hour", "dow"])

    def test_entrenar_modelo_falla_si_menos_de_50_horas(self):
        """Si hay menos de 50 horas de datos, entrenar_modelo retorna None."""
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=30, freq="1h")
        df_corto = pd.DataFrame({
            "datetime": fechas,
            "temp": np.random.uniform(15, 25, 30),
            "PM2_5": np.random.uniform(5, 30, 30),
        })
        metrics, preds, model = entrenar_modelo(df_corto, "PM2_5", features_validas=["temp"])
        self.assertIsNone(metrics)
        self.assertIsNone(preds)
        self.assertIsNone(model)

    def test_generar_features_futuras_con_features_reales(self):
        """Valida que generar_features_futuras solo persista las variables válidas."""
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=10, freq="1h")
        df = pd.DataFrame({
            "datetime": fechas,
            "temp": 22.5,
            "humidity": 55.0,
        })
        future = generar_features_futuras(df, features_validas=["temp", "humidity"], steps=24)
        self.assertIsNotNone(future)
        self.assertEqual(len(future), 24)
        self.assertIn("temp", future.columns)
        self.assertIn("humidity", future.columns)
        self.assertNotIn("pressure", future.columns)

    @patch("core.views.get_mongo_data")
    def test_ejecutar_analisis_sensor_api_exitoso(self, mock_mongo):
        """
        Simula ejecutar_analisis con datos de un sensor tipo API (sin presión ni luz).
        El motor debe entrenar PM2.5 por ML y retornar reporte_limpieza.
        """
        fechas = pd.date_range(start="2026-01-01 00:00:00", periods=70, freq="1h")
        datos_simulados = []
        for f in fechas:
            datos_simulados.append({
                "fecha": f,
                "idDispositivo": "api_sensor_01",
                "temperatura": 20.0 + np.random.normal(0, 1),
                "humedad": 50.0 + np.random.normal(0, 2),
                "presion": None,
                "luz": None,
                "PM2_5": 15.0 + np.random.normal(0, 2),
                "CO2": 500 + int(np.random.normal(0, 20)),
            })
        mock_mongo.return_value = datos_simulados

        resultado = ejecutar_analisis("api_sensor_01")

        self.assertIsNotNone(resultado)
        self.assertIn("reporte_limpieza", resultado)
        self.assertIn("metrics_pm25", resultado)
        # El modelo debió entrenar
        self.assertIsNotNone(resultado["metrics_pm25"])
        self.assertEqual(resultado["metrics_pm25"]["features"], ["temp", "humidity", "hour", "dow"])
        self.assertIn("future_pm25", resultado)
