import math
import os

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Algoritmos de Machine Learning
from sklearn.preprocessing import StandardScaler


class Command(BaseCommand):
    help = (
        "Ejecuta modelos predictivos por sensor en memoria (sin generar archivos CSV)."
    )

    # =========================
    # 1. CONEXIÓN MONGO (Estilo JEJE)
    # =========================
    def cargar_datos_desde_mongo(self):
        cfg = settings.MONGO_SETTINGS

        client = MongoClient(
            host=cfg["HOST"],
            port=cfg["PORT"],
            username=cfg["USERNAME"],
            password=cfg["PASSWORD"],
            authSource=cfg["AUTH_SOURCE"],
            authMechanism=cfg.get("AUTH_MECHANISM", "SCRAM-SHA-1"),
        )

        db = client[cfg["DB_NAME"]]
        # Rescatando colección "lecturas" explícitamente
        collection = db["lecturas"]

        # Proyección optimizada (solo lo necesario)
        cursor = collection.find(
            {},
            {
                "_id": 0,
                "fecha": 1,
                "idDispositivo": 1,
                "temperatura": 1,
                "humedad": 1,
                "presion": 1,
                "luz": 1,
                "PM2_5": 1,
                "CO2": 1,
            },
        )

        df = pd.DataFrame(list(cursor))
        client.close()

        if df.empty:
            return None

        # Normalización de nombres (Estilo JEJE)
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

        # Conversión de fecha robusta
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Conversión numérica
        cols_num = ["temp", "humidity", "pressure", "light", "PM2_5", "CO2"]
        for col in cols_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # =========================
    # 2. FEATURE SELECTION (Estilo JEJE - Dinámico)
    # =========================
    def get_available_features(self, df):
        """Revisa qué columnas existen realmente en este sensor"""
        base_features = ["temp", "humidity", "pressure", "light"]
        return [f for f in base_features if f in df.columns]

    # =========================
    # 3. MOTOR DE ENTRENAMIENTO
    # =========================
    def entrenar_modelo(self, df, target):
        df = df.copy()

        # =========================
        # 1. FEATURES
        # =========================
        env_features = self.get_available_features(df)
        time_features = ["hour", "dow"]
        features = env_features + time_features

        if not env_features:
            return None

        # =========================
        # 2. CREAR FEATURES DE TIEMPO (ANTES)
        # =========================
        if "datetime" in df.columns:
            df["hour"] = df["datetime"].dt.hour
            df["dow"] = df["datetime"].dt.dayofweek

        # =========================
        # 3. LIMPIEZA
        # =========================
        df_clean = df.dropna(subset=["datetime", target] + features)

        if len(df_clean) < 50:
            return None

        # =========================
        # 4. PREPARACIÓN
        # =========================
        X = df_clean[features]
        y = df_clean[target]

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

        # =========================
        # 5. ENTRENAMIENTO
        # =========================
        rf.fit(X, y)
        yhat = rf.predict(X)

        # =========================
        # 6. MÉTRICAS
        # =========================
        return {
            "features": features,
            "R2": r2_score(y, yhat),
            "MAE": mean_absolute_error(y, yhat),
            "RMSE": math.sqrt(mean_squared_error(y, yhat)),
            "n_samples": len(df_clean),
        }

    # =========================
    # 4. EJECUCIÓN (Output estilo pre-PROD / Memoria)
    # =========================
    def handle(self, *args, **options):
        self.stdout.write("📡 Cargando datos desde MongoDB...")
        df = self.cargar_datos_desde_mongo()

        if df is None or df.empty:
            self.stdout.write(self.style.ERROR("❌ No se encontraron datos."))
            return

        # Obtener lista de sensores (Granularidad JEJE)
        devices = sorted(df["idDispositivo"].dropna().unique())
        self.stdout.write(f"🔍 Sensores detectados: {devices}")

        # Diccionario global de resultados (En memoria)
        resultados_globales = {}

        for device_id in devices:
            self.stdout.write(f"\n⚡ Procesando: {device_id}...")
            df_device = df[df["idDispositivo"] == device_id].copy()

            # Resultados del sensor actual
            res_sensor = {}

            # --- Entrenar PM2.5 ---
            metrics_pm25 = self.entrenar_modelo(df_device, "PM2_5")
            if metrics_pm25:
                res_sensor["PM2_5"] = metrics_pm25
                self.stdout.write(
                    f"   ✅ PM2.5 -> R2: {metrics_pm25['R2']:.3f} | Muestras: {metrics_pm25['n_samples']}"
                )
            else:
                self.stdout.write("   ⚠️ PM2.5 -> Datos insuficientes")

            # --- Entrenar CO2 ---
            metrics_co2 = self.entrenar_modelo(df_device, "CO2")
            if metrics_co2:
                res_sensor["CO2"] = metrics_co2
                self.stdout.write(
                    f"   ✅ CO2   -> R2: {metrics_co2['R2']:.3f} | Muestras: {metrics_co2['n_samples']}"
                )
            else:
                self.stdout.write("   ⚠️ CO2   -> Datos insuficientes")

            # Guardamos en memoria global (por si quisieras retornar esto en una vista futura)
            resultados_globales[device_id] = res_sensor

        self.stdout.write(
            self.style.SUCCESS("\n🚀 Proceso finalizado. Todo ejecutado en memoria.")
        )