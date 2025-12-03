# dashboard/management/commands/run_model.py
import os
import math
import pandas as pd
import numpy as np
import shutil
import glob

from django.core.management.base import BaseCommand
from django.conf import settings 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Command(BaseCommand):
    help = 'Ejecuta los modelos predictivos (RL y RF) para PM2.5 o CO2 usando todos los datos.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--target', 
            type=str, 
            default='PM2_5', 
            help='Variable objetivo: PM2_5 o CO2.'
        )

    def handle(self, *args, **options):
        self.stdout.write("⚙️ Iniciando ejecución de modelos sobre todos los datos...")

        INPUT_DIR_BASE = settings.INPUT_DIR_BASE 
        OUT_DIR = settings.OUT_DIR 
        TARGET = options['target']

        # Detectar cualquier CSV/XLSX
        CANDIDATOS = glob.glob(os.path.join(INPUT_DIR_BASE, "*.csv")) + \
                     glob.glob(os.path.join(INPUT_DIR_BASE, "*.xlsx"))

        if not CANDIDATOS:
            self.stderr.write(self.style.ERROR(f"No se encontró ningún archivo CSV o XLSX en: {INPUT_DIR_BASE}"))
            return

        INPUT_PATH = CANDIDATOS[0]
        self.stdout.write(f"📂 Usando archivo de entrada: {INPUT_PATH}")

        os.makedirs(OUT_DIR, exist_ok=True)

        # Detectar separador
        def detectar_separador(path_csv: str) -> str:
            with open(path_csv, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(4096)
            return ";" if head.count(";") >= head.count(",") else ","

        ext = os.path.splitext(INPUT_PATH)[1].lower()
        if ext == ".xlsx":
            df_raw = pd.read_excel(INPUT_PATH, dtype=str)
        else:
            sep = detectar_separador(INPUT_PATH)
            df_raw = pd.read_csv(INPUT_PATH, sep=sep, dtype=str, encoding_errors="replace")

        df_raw.columns = df_raw.columns.str.strip().str.lower()

        # Mapear encabezados
        map_headers = {
            "fecha": "date", "hora": "time", "temperatura": "t", "humedad": "rh",
            "co2": "co2", "pm2_5": "pm2_5", "pm2.5": "pm2_5", "pm25": "pm2_5",
            "presion": "p", "pressure": "p", "luz": "light", "light": "light",
        }

        present = {}
        for col in df_raw.columns:
            if col in map_headers:
                present[map_headers[col]] = col

        required = ["date", "t", "rh", "p", "light", "co2", "pm2_5"]
        missing = [r for r in required if r not in present]
        if missing:
            self.stderr.write(self.style.ERROR(f"Faltan columnas requeridas: {missing}"))
            return

        df = pd.DataFrame({
            "Date": df_raw[present["date"]].astype(str).str.strip(),
            "T": df_raw[present["t"]].astype(str),
            "RH": df_raw[present["rh"]].astype(str),
            "P": df_raw[present["p"]].astype(str), 
            "Light": df_raw[present["light"]].astype(str),
            "CO2": df_raw[present["co2"]].astype(str),
            "PM2_5": df_raw[present["pm2_5"]].astype(str),
        })

        # Limpieza numérica
        def clean_num(s: pd.Series) -> pd.Series:
            s = s.str.strip().str.replace('"', "", regex=False).str.replace("'", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
            s = s.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
            return s

        for col in ["T", "RH", "P", "Light", "CO2", "PM2_5"]:
            df[col] = clean_num(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["datetime"] = pd.to_datetime(df["Date"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["dow"]  = df["datetime"].dt.dayofweek

        features = ["T", "RH", "P", "Light", "hour", "dow"]
        df = df.dropna(subset=["datetime", TARGET])
        mask = df[features].notna().all(axis=1)

        X = df.loc[mask, features].copy()
        y = df.loc[mask, TARGET].copy()
        dt = df.loc[mask, "datetime"].copy()

        self.stdout.write(f"Registros utilizables para modelado: {len(X)}")

        # Pipelines
        num_cols = features
        pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")

        linreg = Pipeline([("pre", pre), ("model", LinearRegression())])
        rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1))])

        # Entrenamiento sobre todo el dataset
        self.stdout.write("🧠 Entrenando modelos...")
        linreg.fit(X, y)
        rf.fit(X, y)

        # Predicciones sobre todo el dataset
        yhat_lin = linreg.predict(X)
        yhat_rf  = rf.predict(X)

        # Guardar predicciones
        if TARGET == "PM2_5":
            out_pred = pd.DataFrame({
                "datetime": dt.values,
                "pm25_valor_real": y.values,
                "pm25_pred_reg_lineal": yhat_lin,
                "pm25_pred_randomforest": yhat_rf
            }).sort_values("datetime")
        elif TARGET == "CO2":
            out_pred = pd.DataFrame({
                "datetime": dt.values,
                "co2_valor_real": y.values,
                "co2_pred_reg_lineal": yhat_lin,
                "co2_pred_randomforest": yhat_rf
            }).sort_values("datetime")

        pred_filename = os.path.join(OUT_DIR, f"predicciones_{TARGET}.csv")
        out_pred.to_csv(pred_filename, index=False)
        self.stdout.write(f"Predicciones guardadas en CSV: {pred_filename}")

        # Gráfica
        plt.figure(figsize=(10,5))
        if TARGET == "PM2_5":
            plt.plot(out_pred["datetime"], out_pred["pm25_valor_real"], label="Real")
            plt.plot(out_pred["datetime"], out_pred["pm25_pred_reg_lineal"], label="Regresión Lineal", alpha=0.8)
            plt.plot(out_pred["datetime"], out_pred["pm25_pred_randomforest"], label="Random Forest", alpha=0.8)
        else:
            plt.plot(out_pred["datetime"], out_pred["co2_valor_real"], label="Real")
            plt.plot(out_pred["datetime"], out_pred["co2_pred_reg_lineal"], label="Regresión Lineal", alpha=0.8)
            plt.plot(out_pred["datetime"], out_pred["co2_pred_randomforest"], label="Random Forest", alpha=0.8)

        plt.title(f"Predicción de {TARGET}")
        plt.xlabel("Tiempo"); plt.ylabel(TARGET)
        plt.legend(); plt.tight_layout()

        png_filename = os.path.join(OUT_DIR, f"pred_vs_real_{TARGET}.png")
        plt.savefig(png_filename, dpi=150)
        STATIC_APP_DIR = os.path.join(settings.BASE_DIR, 'core', 'static', 'dashboard')
        os.makedirs(STATIC_APP_DIR, exist_ok=True)
        shutil.copy(png_filename, os.path.join(STATIC_APP_DIR, f"pred_vs_real_{TARGET}.png"))
        plt.close()
        self.stdout.write(f"Gráfica guardada: {png_filename}")

        # Métricas globales
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics_lin = {
            "MAE": mean_absolute_error(y, yhat_lin),
            "RMSE": math.sqrt(mean_squared_error(y, yhat_lin)),
            "R2": r2_score(y, yhat_lin)
        }
        metrics_rf = {
            "MAE": mean_absolute_error(y, yhat_rf),
            "RMSE": math.sqrt(mean_squared_error(y, yhat_rf)),
            "R2": r2_score(y, yhat_rf)
        }

        metrics_df = pd.DataFrame([{
            "MAE_lin": metrics_lin["MAE"],
            "RMSE_lin": metrics_lin["RMSE"],
            "R2_lin": metrics_lin["R2"],
            "MAE_rf": metrics_rf["MAE"],
            "RMSE_rf": metrics_rf["RMSE"],
            "R2_rf": metrics_rf["R2"]
        }])
        metrics_csv_path = os.path.join(OUT_DIR, f"metrics_{TARGET}.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        self.stdout.write(f"Métricas globales guardadas en CSV: {metrics_csv_path}")

        self.stdout.write(self.style.SUCCESS(f"✅ Modelos de {TARGET} ejecutados sobre todos los registros."))



""" # dashboard/management/commands/run_model.py
import os
import math
import pandas as pd
import numpy as np
import shutil

# Importaciones específicas de Django
from django.core.management.base import BaseCommand
from django.conf import settings 

# Para evitar problemas de Tk/tcl en Windows al graficar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Importaciones de Machine Learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ------------------------------------------------------------------
# CLASE DEL COMANDO DE GESTIÓN
# ------------------------------------------------------------------

class Command(BaseCommand):
    help = 'Ejecuta los modelos predictivos (RL y RF) para PM2.5 o CO2.'

    def add_arguments(self, parser):
        # Permite al usuario especificar la variable objetivo: python manage.py run_model --target CO2
        parser.add_argument(
            '--target', 
            type=str, 
            default='PM2_5', 
            help='Variable objetivo: PM2_5 o CO2.'
        )

    def handle(self, *args, **options):
        self.stdout.write("⚙️ Iniciando ejecución de modelos...")

        # ------------ Configuración (Adaptada a Django Settings) ------------
        
        # Obtener las rutas portables desde settings.py
        INPUT_DIR_BASE = settings.INPUT_DIR_BASE 
        OUT_DIR = settings.OUT_DIR 
        
        # Variable objetivo desde los argumentos del comando
        TARGET = options['target'] 
        
        # Nombres candidatos (misma base, distintas extensiones)
        CANDIDATOS = [
             os.path.join(INPUT_DIR_BASE, "datosEstacioITMfinal_model.csv"),
             os.path.join(INPUT_DIR_BASE, "datosEstacioITMfinal_model.xlsx"),
        ]
        
        # Asegurarse de que el directorio de salida exista
        os.makedirs(OUT_DIR, exist_ok=True)
        
        # ------------ Carga robusta (EL RESTO DEL SCRIPT ORIGINAL) ------------
        def detectar_separador(path_csv: str) -> str:
            with open(path_csv, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(4096)
            return ";" if head.count(";") >= head.count(",") else ","


        def cargar_entrada():
            input_path = None
            for p in CANDIDATOS:
                if os.path.exists(p):
                    input_path = p
                    break
            if input_path is None:
                raise FileNotFoundError(
                    f"No se encontró ninguno de los archivos candidatos:\n- " +
                    "\n- ".join(CANDIDATOS) +
                    "\nColoca el archivo con ese nombre (CSV o XLSX) en la carpeta indicada."
                )

            self.stdout.write(f"📂 Usando archivo de entrada: {input_path}")
            
            ext = os.path.splitext(input_path)[1].lower()
            if ext == ".xlsx":
                df_raw = pd.read_excel(input_path, dtype=str)
            else:
                sep = detectar_separador(input_path)
                df_raw = pd.read_csv(input_path, sep=sep, dtype=str, encoding_errors="replace")

            return input_path, df_raw


        try:
            INPUT_PATH, df_raw = cargar_entrada()
        except FileNotFoundError as e:
            self.stderr.write(self.style.ERROR(str(e)))
            return
            
        # Normalizar encabezados: minúsculas, sin espacios
        df_raw.columns = df_raw.columns.str.strip().str.lower()

        # Mapeo de encabezados: origen -> destino interno
        map_headers = {
            "fecha": "date", "hora": "time", "temperatura": "t", "humedad": "rh",
            "co2": "co2", "pm2_5": "pm2_5", "pm2.5": "pm2_5", "pm25": "pm2_5",
            "presion": "p", "pressure": "p", "luz": "light", "light": "light",
        }

        # Ver qué columnas equivalentes están presentes
        present = {}
        for col in df_raw.columns:
            if col in map_headers:
                present[map_headers[col]] = col

        # Requeridas
        required = ["date", "time", "t", "rh", "p", "light", "co2", "pm2_5"]
        missing = [r for r in required if r not in present]
        if missing:
            self.stderr.write(self.style.ERROR(
                f"Faltan columnas requeridas: {missing}"
            ))
            return

        # Construir DataFrame con nombres homogéneos (inglés interno)
        df = pd.DataFrame({
            "Date": df_raw[present["date"]].astype(str).str.strip(),
            "Time": df_raw[present["time"]].astype(str).str.strip(),
            "T": df_raw[present["t"]].astype(str),
            "RH": df_raw[present["rh"]].astype(str),
            "P": df_raw[present["p"]].astype(str),
            "Light": df_raw[present["light"]].astype(str),
            "CO2": df_raw[present["co2"]].astype(str),
            "PM2_5": df_raw[present["pm2_5"]].astype(str),
        })


        # ------------ Limpieza de cadenas y numéricos ------------
        def clean_num(s: pd.Series) -> pd.Series:
            s = s.str.strip().str.replace('"', "", regex=False).str.replace("'", "", regex=False)
            s = s.str.replace(",", ".", regex=False)  # coma decimal -> punto
            s = s.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
            return s

        for col in ["T", "RH", "P", "Light", "CO2", "PM2_5"]:
            df[col] = clean_num(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Date"] = df["Date"].str.replace('"', "", regex=False).str.strip()
        df["Time"] = df["Time"].str.replace('"', "", regex=False).str.strip()
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")

        # Features temporales
        df["hour"] = df["datetime"].dt.hour
        df["dow"]  = df["datetime"].dt.dayofweek  # 0=Lunes..6=Domingo

        # Diagnóstico antes de filtrar (opcional)
        # self.stdout.write("---- Diagnóstico de NaN antes de filtrar ----")
        # self.stdout.write(str(df[["T","RH","P","Light","CO2","PM2_5","datetime"]].isna().sum()))

        # ------------ Selección de variables ------------
        features = ["T", "RH", "P", "Light", "hour", "dow"]

        # Quitar filas inválidas: sin datetime, sin TARGET, o con NaN en features
        df = df.dropna(subset=["datetime", TARGET])
        mask = df[features].notna().all(axis=1)

        X = df.loc[mask, features].copy()
        y = df.loc[mask, TARGET].copy()
        dt = df.loc[mask, "datetime"].copy()

        self.stdout.write(f"Registros utilizables para modelado: {len(X)}")

        if len(X) < 50:
            self.stderr.write(self.style.ERROR(f"Quedan muy pocos registros ({len(X)}) después de limpiar. Revisa los datos."))
            return


        # ------------ Split train/test ------------
        X_train, X_test, y_train, y_test, dt_train, dt_test = train_test_split(
            X, y, dt, test_size=0.2, random_state=42
        )

        # ------------ Pipelines ------------
        num_cols = ["T", "RH", "P", "Light", "hour", "dow"]

        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(), num_cols)],
            remainder="drop"
        )

        linreg = Pipeline(steps=[
            ("pre", pre),
            ("model", LinearRegression())
        ])

        rf = Pipeline(steps=[
            ("pre", pre),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # ------------ Entrenamiento ------------
        self.stdout.write("🧠 Entrenando modelos...")
        linreg.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # ------------ Evaluación ------------
        def metrics(model, Xtr, ytr, Xte, yte):
            pred_tr = model.predict(Xtr)
            pred_te = model.predict(Xte)
            mae_tr = mean_absolute_error(ytr, pred_tr)
            rmse_tr = math.sqrt(mean_squared_error(ytr, pred_tr))
            r2_tr = r2_score(ytr, pred_tr)
            mae_te = mean_absolute_error(yte, pred_te)
            rmse_te = math.sqrt(mean_squared_error(yte, pred_te))
            r2_te = r2_score(yte, pred_te)
            return {
                "MAE_train": mae_tr, "RMSE_train": rmse_tr, "R2_train": r2_tr,
                "MAE_test": mae_te, "RMSE_test": rmse_te, "R2_test": r2_te
            }, pred_te

        m_lin, yhat_lin = metrics(linreg, X_train, y_train, X_test, y_test)
        m_rf,  yhat_rf  = metrics(rf,      X_train, y_train, X_test, y_test)

        # ------------ Validación cruzada ------------
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer_r2 = make_scorer(r2_score)
        cv_r2_lin = cross_val_score(linreg, X, y, cv=cv, scoring=scorer_r2)
        cv_r2_rf  = cross_val_score(rf, X, y, cv=cv, scoring=scorer_r2)

        # ------------ Guardar métricas ------------
        def fmt_metrics(title, m, cv_r2):
            s = [f"== {title} =="]
            s.append(f"MAE  (train/test): {m['MAE_train']:.3f} / {m['MAE_test']:.3f}")
            s.append(f"RMSE (train/test): {m['RMSE_train']:.3f} / {m['RMSE_test']:.3f}")
            s.append(f"R2   (train/test): {m['R2_train']:.3f} / {m['R2_test']:.3f}")
            s.append(f"CV R2 (5-fold): mean={cv_r2.mean():.3f}, std={cv_r2.std():.3f}, scores={np.round(cv_r2,3)}")
            return "\n".join(s)

        report = []
        report.append(f"ARCHIVO ENTRADA: {INPUT_PATH}")
        report.append(f"VARIABLE OBJETIVO: {TARGET}\n")
        report.append(fmt_metrics("Regresión Lineal", m_lin, cv_r2_lin))
        report.append("")
        report.append(fmt_metrics("Random Forest", m_rf, cv_r2_rf))
        report_txt = "\n\n".join(report)

        # Nombre del archivo de métricas con el TARGET actual
        metrics_filename = os.path.join(OUT_DIR, f"metricas_{TARGET}.txt")
        with open(metrics_filename, "w", encoding="utf-8") as f:
            f.write(report_txt)

        # ------------ Guardar predicciones ------------
        out_pred = pd.DataFrame({
            "datetime": dt_test.values,
            f"y_true_{TARGET}": y_test.values,
            f"yhat_lin_{TARGET}": yhat_lin,
            f"yhat_rf_{TARGET}": yhat_rf
        }).sort_values("datetime")

        # Nombre del archivo de predicciones con el TARGET actual
        pred_filename = os.path.join(OUT_DIR, f"predicciones_{TARGET}.csv")
        out_pred.to_csv(pred_filename, index=False)

        # ------------ Gráfica (guardada) ------------
        plt.figure(figsize=(10,5))
        plt.plot(out_pred["datetime"], out_pred[f"y_true_{TARGET}"], label="Real")
        plt.plot(out_pred["datetime"], out_pred[f"yhat_lin_{TARGET}"], label="LinReg", alpha=0.8)
        plt.plot(out_pred["datetime"], out_pred[f"yhat_rf_{TARGET}"], label="RandomForest", alpha=0.8)
        plt.title(f"Predicción de {TARGET}")
        plt.xlabel("Tiempo"); plt.ylabel(TARGET)
        plt.legend()
        plt.tight_layout()
        
        # Nombre del archivo PNG con el TARGET actual
        png_filename = os.path.join(OUT_DIR, f"pred_vs_real_{TARGET}.png")
        plt.savefig(png_filename, dpi=150)
        # Ruta donde se guardan las estáticas de la app
        STATIC_APP_DIR = os.path.join(settings.BASE_DIR, 'core', 'static', 'dashboard')
        os.makedirs(STATIC_APP_DIR, exist_ok=True) # Asegura que la carpeta exista

        # Ruta de destino del PNG (para que Django lo sirva)
        TARGET_PNG_PATH = os.path.join(STATIC_APP_DIR, f"pred_vs_real_{TARGET}.png")

        # Copiar el PNG generado a la carpeta estática
        shutil.copy(png_filename, TARGET_PNG_PATH) 

        self.stdout.write(f"🖼️ Gráfica copiada a estáticos para web: {TARGET_PNG_PATH}")
        plt.close()

        self.stdout.write(self.style.SUCCESS(f'\n✅ Modelos de {TARGET} ejecutados con éxito.'))
        self.stdout.write(f"Resultados guardados en: {metrics_filename}")
        self.stdout.write(f"Predicciones guardadas en: {pred_filename}")
        self.stdout.write(f"Gráfica guardada en: {png_filename}") """