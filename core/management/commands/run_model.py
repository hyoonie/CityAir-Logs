import math
import os

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from pymongo import MongoClient
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


class Command(BaseCommand):
    help = (
        "Ejecuta modelos predictivos por sensor en memoria usando el pipeline de limpieza centralizado (Fase A)."
    )

    # =========================
    # 1. CONEXIÓN MONGO
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
        collection = db["lecturas"]

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

        return df

    # =========================
    # 2. MOTOR DE ENTRENAMIENTO
    # =========================
    def entrenar_modelo(self, df, target, features_validas=None):
        if "datetime" not in df.columns:
            return None

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Variables temporales
        df["hour"] = df["datetime"].dt.hour
        df["dow"] = df["datetime"].dt.dayofweek

        if features_validas is not None:
            env_features = [f for f in features_validas if f in df.columns]
        else:
            env_features = [f for f in COLUMNAS_AMBIENTALES if f in df.columns]

        time_features = ["hour", "dow"]
        features = env_features + time_features

        if not env_features:
            return None

        required_cols = ["datetime", target] + features
        df_clean = df.dropna(subset=required_cols)

        if len(df_clean) < MIN_HORAS_ENTRENAMIENTO:
            return None

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

        rf.fit(X, y)
        yhat = rf.predict(X)

        return {
            "features": features,
            "R2": r2_score(y, yhat),
            "MAE": mean_absolute_error(y, yhat),
            "RMSE": math.sqrt(mean_squared_error(y, yhat)),
            "n_samples": len(df_clean),
        }

    # =========================
    # 3. EJECUCIÓN
    # =========================
    def handle(self, *args, **options):
        self.stdout.write("📡 Cargando datos desde MongoDB...")
        df_mongo = self.cargar_datos_desde_mongo()

        if df_mongo is None or df_mongo.empty:
            self.stdout.write(self.style.ERROR("❌ No se encontraron datos."))
            return

        # Normalizar identificador de dispositivo
        id_col = "idDispositivo" if "idDispositivo" in df_mongo.columns else "iddispositivo"
        if id_col not in df_mongo.columns:
            self.stdout.write(self.style.ERROR("❌ No se encontró columna de dispositivo."))
            return

        devices = sorted(df_mongo[id_col].dropna().unique())
        self.stdout.write(f"🔍 Sensores detectados: {devices}")

        resultados_globales = {}

        for device_id in devices:
            self.stdout.write(f"\n⚡ Procesando: {device_id}...")
            df_device_raw = df_mongo[df_mongo[id_col] == device_id].copy()

            # Aplicar pipeline centralizado de limpieza (Fase A)
            resultado = limpiar_lecturas(df_device_raw, targets=["PM2_5", "CO2"])
            df_device = resultado.df
            features_validas = resultado.features_validas
            targets_validos = resultado.targets_validos
            reporte = resultado.reporte

            self.stdout.write(
                f"   📊 Horas limpias: {reporte['horas_con_datos']}/{reporte['horas_totales_periodo']} | "
                f"Features válidas (>=60%): {features_validas}"
            )

            res_sensor = {"reporte": reporte}

            # --- Entrenar PM2.5 ---
            if "PM2_5" in targets_validos:
                metrics_pm25 = self.entrenar_modelo(df_device, "PM2_5", features_validas=features_validas)
                if metrics_pm25:
                    res_sensor["PM2_5"] = metrics_pm25
                    self.stdout.write(
                        f"   ✅ PM2.5 -> R2: {metrics_pm25['R2']:.3f} | Muestras: {metrics_pm25['n_samples']} | "
                        f"Features: {metrics_pm25['features']}"
                    )
                else:
                    motivo = f"Horas válidas con datos completos insuficientes (< {MIN_HORAS_ENTRENAMIENTO} h)."
                    reporte["motivo_fallback"]["PM2_5"] = motivo
                    self.stdout.write(f"   ⚠️ PM2.5 -> {motivo}")
            else:
                motivo = reporte["motivo_fallback"].get("PM2_5", "Cobertura < 60%")
                self.stdout.write(f"   ⚠️ PM2.5 -> Fallback: {motivo}")

            # --- Entrenar CO2 ---
            if "CO2" in targets_validos:
                metrics_co2 = self.entrenar_modelo(df_device, "CO2", features_validas=features_validas)
                if metrics_co2:
                    res_sensor["CO2"] = metrics_co2
                    self.stdout.write(
                        f"   ✅ CO2   -> R2: {metrics_co2['R2']:.3f} | Muestras: {metrics_co2['n_samples']} | "
                        f"Features: {metrics_co2['features']}"
                    )
                else:
                    motivo = f"Horas válidas con datos completos insuficientes (< {MIN_HORAS_ENTRENAMIENTO} h)."
                    reporte["motivo_fallback"]["CO2"] = motivo
                    self.stdout.write(f"   ⚠️ CO2   -> {motivo}")
            else:
                motivo = reporte["motivo_fallback"].get("CO2", "Cobertura < 60%")
                self.stdout.write(f"   ⚠️ CO2   -> Fallback: {motivo}")

            resultados_globales[device_id] = res_sensor

        self.stdout.write(
            self.style.SUCCESS("\n🚀 Proceso finalizado. Todo ejecutado en memoria con pipeline limpio.")
        )
