import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# FEATURES DISPONIBLES
# =========================
def get_available_features(df):
    base = ["temp", "humidity", "pressure", "light"]
    return [c for c in base if c in df.columns]


# =========================
# ENTRENAMIENTO
# =========================
def entrenar_modelo(df, target):
    """
    Entrena modelo en memoria de forma segura.
    Crea variables temporales antes de limpiar.
    """

    # -------------------------
    # 1. Normalización tiempo
    # -------------------------
    if "datetime" not in df.columns:
        return None, None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # variables temporales ANTES de limpiar
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek

    # -------------------------
    # 2. Features dinámicos
    # -------------------------
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

    return metrics, preds
