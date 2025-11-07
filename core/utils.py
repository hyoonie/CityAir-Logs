# core/utils.py
import os
import re
from django.conf import settings


def _load_model_kpis(target='PM2_5'):
    """
    Carga las métricas (MAE, RMSE, R2) y la ruta de la imagen generadas por el modelo 
    desde el archivo TXT y define el nombre del archivo PNG estático.
    """
    
    # ¡¡ASEGÚRATE DE DEFINIR OUT_DIR EN settings.py!!
    # (settings.py) OUT_DIR = os.path.join(settings.BASE_DIR, 'data_output')
    OUT_DIR = getattr(settings, 'OUT_DIR', '') 
    
    # 1. Definir rutas de los archivos generados
    PNG_FILENAME = f"pred_vs_real_{target}.png"
    METRICS_FILENAME = f"metricas_{target}.txt"
    
    METRICS_PATH_FULL = os.path.join(OUT_DIR, METRICS_FILENAME)
    
    # 2. Valores por defecto
    data = {
        "MAE": "N/A", 
        "RMSE": "N/A", 
        "R2": "N/A", 
        "png_filename": PNG_FILENAME, # El nombre del archivo que JS buscará en static/dashboard/
        "error_msg": None,
        "is_data_loaded": False
    }

    try:
        # 3. Leer Métricas del archivo TXT
        if os.path.exists(METRICS_PATH_FULL):
            with open(METRICS_PATH_FULL, 'r', encoding='utf-8') as f:
                report = f.read()
                
            # Extraer valores de prueba (test) del informe de Random Forest usando expresiones regulares
            mae_match = re.search(r'== Random Forest ==.*?MAE\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            rmse_match = re.search(r'== Random Forest ==.*?RMSE\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            r2_match = re.search(r'== Random Forest ==.*?R2\s+\(train/test\):\s+[\d.]+?\s+/\s+([\d.]+)', report, re.DOTALL)
            
            if r2_match and mae_match and rmse_match:
                data['R2'] = f"{float(r2_match.group(1)):.3f}"
                data['MAE'] = f"{float(mae_match.group(1)):.3f}"
                data['RMSE'] = f"{float(rmse_match.group(1)):.3f}"
                data['is_data_loaded'] = True
            else:
                data["error_msg"] = "Archivo de métricas encontrado, pero el formato no es el esperado (R2/MAE/RMSE)."
        else:
            data["error_msg"] = f"Archivos de modelo para {target} no encontrados. Ejecuta el Generador de Modelos."
        
    except Exception as e:
        data["error_msg"] = f"Error general al cargar KPIs: {e}"
        
    return data