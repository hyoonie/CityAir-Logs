import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def ejecutar_modelo(df, target="PM2_5"):
    """
    Entrena LinearRegression y RandomForest sobre df y devuelve:
    - DataFrame de predicciones
    - Diccionario de métricas
    - Figura Matplotlib como objeto
    """
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()
    map_headers = {
        "fecha":"date", "hora":"time", "temperatura":"t", "humedad":"rh",
        "co2":"co2", "pm2_5":"pm2_5", "pm2.5":"pm2_5", "pm25":"pm2_5",
        "presion":"p", "pressure":"p", "luz":"light", "light":"light"
    }

    # Detectar columnas presentes
    present = {map_headers[c]: c for c in df.columns if c in map_headers}

    # Requeridas (sin "time" obligatorio)
    required = ["date", "t", "rh", "p", "light", "co2", "pm2_5"]
    for r in required:
        if r not in present:
            raise ValueError(f"Falta columna requerida: {r}")

    # Construir DataFrame
    df_proc = pd.DataFrame({
        "Date": df[present["date"]].astype(str).str.strip(),
        "T": df[present["t"]].astype(float),
        "RH": df[present["rh"]].astype(float),
        "P": df[present["p"]].astype(float),
        "Light": df[present["light"]].astype(float),
        "CO2": df[present["co2"]].astype(float),
        "PM2_5": df[present["pm2_5"]].astype(float)
    })

    # Extraer datetime de la columna 'fecha'
    df_proc["datetime"] = pd.to_datetime(df_proc["Date"], errors="coerce")
    df_proc["hour"] = df_proc["datetime"].dt.hour
    df_proc["dow"]  = df_proc["datetime"].dt.dayofweek

    features = ["T", "RH", "P", "Light", "hour", "dow"]

    # Filtrar filas con NaN
    df_proc = df_proc.dropna(subset=["datetime", target])
    mask = df_proc[features].notna().all(axis=1)
    X = df_proc.loc[mask, features]
    y = df_proc.loc[mask, target]
    dt = df_proc.loc[mask, "datetime"]

    # Split train/test
    X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
        X, y, dt, test_size=0.2, random_state=42
    )

    # Pipelines
    pre = ColumnTransformer([("num", StandardScaler(), features)], remainder="drop")
    linreg = Pipeline([("pre", pre), ("model", LinearRegression())])
    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])

    # Entrenamiento
    linreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Métricas
    def metrics(model, Xtr, ytr, Xte, yte):
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
        mae_tr = mean_absolute_error(ytr, pred_tr)
        rmse_tr = math.sqrt(mean_squared_error(ytr, pred_tr))
        r2_tr = r2_score(ytr, pred_tr)
        mae_te = mean_absolute_error(yte, pred_te)
        rmse_te = math.sqrt(mean_squared_error(yte, pred_te))
        r2_te = r2_score(yte, pred_te)
        return {"MAE_train": mae_tr, "RMSE_train": rmse_tr, "R2_train": r2_tr,
                "MAE_test": mae_te, "RMSE_test": rmse_te, "R2_test": r2_te}, pred_te

    m_lin, yhat_lin = metrics(linreg, X_train, y_train, X_test, y_test)
    m_rf,  yhat_rf  = metrics(rf, X_train, y_train, X_test, y_test)

    # Validación cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer_r2 = make_scorer(r2_score)
    cv_r2_lin = cross_val_score(linreg, X, y, cv=cv, scoring=scorer_r2)
    cv_r2_rf  = cross_val_score(rf, X, y, cv=cv, scoring=scorer_r2)

    # DataFrame de predicciones
    out_pred = pd.DataFrame({
        "datetime": dt_test.values,
        f"y_true_{target}": y_test.values,
        f"yhat_lin_{target}": yhat_lin,
        f"yhat_rf_{target}": yhat_rf
    }).sort_values("datetime")

    # Figura
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(out_pred["datetime"], out_pred[f"y_true_{target}"], label="Real")
    ax.plot(out_pred["datetime"], out_pred[f"yhat_lin_{target}"], label="LinReg", alpha=0.8)
    ax.plot(out_pred["datetime"], out_pred[f"yhat_rf_{target}"], label="RandomForest", alpha=0.8)
    ax.set_title(f"Predicción de {target}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel(target)
    ax.legend()
    plt.tight_layout()

    metrics_dict = {
        "linreg": m_lin, "rf": m_rf,
        "cv_r2_lin": cv_r2_lin, "cv_r2_rf": cv_r2_rf
    }

    # al final de ejecutar_modelo
    plt.tight_layout()
    fig.savefig("alguna_ruta_opcional.png")  # si quieres guardar
    plt.close(fig)  # cierra la figura antes de salir
    
    return out_pred, metrics_dict, fig

