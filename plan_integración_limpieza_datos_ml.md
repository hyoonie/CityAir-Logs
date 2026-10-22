# Plan: integrar la limpieza de datos previa al entrenamiento (RandomForest)

Estado: **borrador para revisión**. No hay código escrito. Los rangos físicos (P1) ya tienen propuesta en la sección 3.1, con tres confirmaciones pendientes; la cobertura mínima (P2) sigue pendiente. El resto del plan no depende de ellos.

## 1. Contexto

La colección `lecturas` de Mongo recibe documentos de tres fuentes (TTN, Redspira, API de sensores) con esquemas, tipos y nulos distintos (ver la memoria `project-ingesta-datos-mongo`). El motor predictivo ([core/management/commands/run_engine.py](core/management/commands/run_engine.py)) y el comando de diagnóstico ([core/management/commands/run_model.py](core/management/commands/run_model.py)) entrenan un `RandomForestRegressor` para `PM2_5` y `CO2` con una limpieza mínima.

**Decisión de ubicación (confirmada):** la limpieza se aplica **al leer**, antes de entrenar. Mongo conserva el dato crudo. No se toca la ingesta (listener, worker, API) ni el historial.

**Hay dos consumidores de los datos, no uno:**
- **Fase A, predicción:** `run_engine.py` y `run_model.py` (RandomForest).
- **Fase B, clustering y gráficas:** la vista de análisis de datos ([views.py:160-480](core/views.py#L160-L480)), que usa KMeans (perfiles de contaminación), DBSCAN (momentos atípicos), un puntaje DEA y las gráficas por dispositivo.

## 2. Hallazgos que justifican el plan

| # | Hallazgo | Evidencia |
|---|---|---|
| H1 | Las variables del modelo se eligen por **existencia de columna**, no por datos. Un dispositivo de la API trae `presion` y `luz` siempre en `None`: el `dropna` deja 0 de 120 filas y el motor cae al plan B plano (promedio de 24 lecturas, sin R² ni resumen). | Simulación con datos sintéticos; [run_engine.py:39-58](core/management/commands/run_engine.py#L39-L58) |
| H2 | Se entrena con lecturas crudas y se predice en pasos de **una hora**. Las fuentes tienen frecuencias distintas y Redspira puede repetir el mismo valor muchas veces (por verificar con datos reales). | [run_engine.py:148-152](core/management/commands/run_engine.py#L148-L152), [redspira_worker.py:57-60](core/management/commands/redspira_worker.py#L57-L60) |
| H3 | No hay validación de rango físico: un `-999`, un `0` de sensor desconectado o una humedad de 150 entran tal cual. | No existe en ningún módulo |
| H4 | Tipos mezclados en `CO2` y demás (la API usa `int()`/`float()`; TTN y Redspira guardan lo que reciben). Hoy se resuelve con `to_numeric(coerce)`. | [views.py:1099](core/views.py#L1099) |
| H5 | La limpieza está duplicada en `run_engine.py`, `run_model.py` y la vista de análisis de datos, y ya divergen. | Los tres sitios |
| H6 | El clustering (KMeans, DBSCAN) lee los mismos datos crudos: variables `PM2_5`, `CO2`, `temperatura`, `humedad` y `luz` elegidas por existencia de columna, sin validación de rango. | [views.py:246-248](core/views.py#L246-L248) |
| H7 | `StandardScaler` es sensible a valores imposibles: un `-999` desplaza media y desviación y comprime el resto. DBSCAN (`eps=0.5`, `min_samples=5`) lo reporta como "momento atípico" junto con los eventos reales. | [views.py:359-429](core/views.py#L359-L429) |
| H8 | Los huecos se rellenan con `bfill().fillna(0)`: cada hueco toma el valor de la hora **siguiente** (usa datos futuros) y los huecos del final, o una columna vacía, pasan a **0**, un valor falso que el escalador trata como real. | [views.py:357](core/views.py#L357) |
| H9 | Con varios dispositivos (hasta 3), el remuestreo promedia todos juntos antes de agrupar. Además, el reporte de calidad solo cuenta filas **totalmente** vacías: la "integridad" y su texto "Excelente" no miden nulos ni valores inválidos. | [views.py:273-300](core/views.py#L273-L300) |

## 3. Parámetros pendientes (los define el usuario)

Se guardan como constantes en un solo lugar del módulo nuevo, para cambiarlos sin tocar la lógica.

| Parámetro | Estado | Nota |
|---|---|---|
| **P1. Rangos físicos** por variable (mín y máx de validez) | **Propuesta en 3.1; falta `luz`, y confirmar presión y CO2** | Variables: `PM2_5`, `PM10`, `CO2`, `temperatura`, `humedad`, `presion`, `luz`. Hay que confirmar la **unidad** de `presion` y `luz`: TTN y Redspira no la documentan en el repo. |
| **P2. Cobertura mínima** para usar una variable | **Pendiente del usuario** | Porcentaje de horas con dato en el periodo consultado (ejemplo a discutir: 60 %). |
| P3. Mínimo de horas para entrenar | Propuesta: conservar 50, ahora medidas en **horas** | Hoy son 50 filas crudas; tras remuestrear equivale a 50 horas (~2 días). A confirmar. |
| P4. Huecos en variables (predicción y clustering) | Propuesta: **no imputar**; se descartan las horas incompletas | Alternativa: `ffill` limitado a 2 horas. A confirmar. En la Fase B reemplaza a `bfill().fillna(0)`. |
| P5. Definición de "integridad" en el reporte de calidad (Fase B) | Propuesta: % de valores válidos sobre los esperados en el periodo | Hoy mide solo filas totalmente vacías (H9). A confirmar. |

Mientras P1 (completo y confirmado) y P2 no estén definidos, la implementación no inicia.

### 3.1 Propuesta de P1: rangos físicos

Referencias: SEDEMA Veracruz y NOM-025-SSA1-2021 (calidad del aire); SMN/CONAGUA (clima). Xalapa está a ~1,420 msnm, clima templado húmedo.

Cada variable tiene **dos capas** que no deben confundirse:

- **Rango de validez (el que usa el código):** amplio. Solo anula lo imposible o una falla de sensor.
- **Referencia climatológica (informativa):** lo normal y lo extremo en Xalapa. No se usa para recortar, porque el plan conserva los picos plausibles (paso 4 de 4.1) y un corte en 150 µg/m³ borraría eventos reales de pirotecnia o incendios.

| Variable | Unidad | Rango de validez | Referencia climatológica (informativa) | Estado |
|---|---|---|---|---|
| `PM2_5` | µg/m³ | 0 – 1000 | Aire limpio ~2–5; eventos críticos >150; norma diaria 33 (promedio 24 h) | Propuesto |
| `PM10` | µg/m³ | 0 – 1000 | Aire limpio ~5–10; tolvaneras o quemas >250; norma diaria 60 | Propuesto |
| `CO2` | ppm | 300 – 5000 | Aire libre: fondo global ~415–420; rango de referencia 380–2000; >1000 solo en interiores mal ventilados | Propuesto (ver notas) |
| `temperatura` | °C | 0 – 50 | Mínimo histórico 1.7; invierno común 5–8; máximo reportado 42.5 | Propuesto (ver notas) |
| `humedad` | % | 0 – 100 | ~15–25 en "surada" seca; hasta 100 con niebla o lluvia | Propuesto |
| `presion` | hPa | 800 – 900 | Presión de estación ~850–870 a 1,420 msnm | **Confirmar unidad y tipo** (ver notas) |
| `luz` | **Falta** | **Falta** | No vino en la lista | **Pendiente del usuario** |

Notas:

- **Presión:** el máximo de 1016 hPa de la lista corresponde a presión a nivel del mar; a la altitud de Xalapa la presión de estación rara vez pasa de ~870 hPa. Si el sensor reporta presión de estación, el rango propuesto aplica. Si reporta presión reducida a nivel del mar, el rango sería ~980–1040. Hay que confirmar cuál entrega el dispositivo y en qué unidad (hPa, Pa o kPa).
- **Temperatura:** el máximo de 42.5 °C (mayo de 2025) no lo pude verificar y me parece alto para Xalapa; conviene contrastarlo con el SMN. Además un sensor expuesto al sol marca más que el aire, así que el corte se deja en 50 °C y no en el récord oficial.
- **CO2:** el rango 380–2000 ppm queda como referencia, no como corte. Un sensor NDIR o electroquímico puede marcar por debajo de 380 por deriva o calibración, y un pico real sobre 2000 (cerca de una fuente de combustión o en interior) se anularía. Por eso el corte propuesto es 300–5000, que solo atrapa fallas. Si prefieres usar 380–2000 como corte, es cambiar dos constantes, pero se perderían esas lecturas legítimas. La fuente citada (ASHRAE) es una guía de ventilación interior, y NOM-011-STPS trata ruido, no CO2, así que conviene revisar esa referencia.
- **Rangos de exterior:** la lista describe condiciones al aire libre. Si algún dispositivo está en interior, `CO2` y temperatura necesitan un rango distinto (con CO2 en interior, 2000 ppm queda corto como referencia).
- **Ceros exactos:** el hallazgo H3 menciona el `0` de un sensor desconectado. Falta decidir si un `0` exacto en `PM2_5`, `PM10` y `CO2` se trata como falla (propuesta: sí para `CO2`, que nunca es 0 en el aire; no para PM, donde 0 es poco común pero posible).
- La lista original repetía el bloque de PM10 donde probablemente iba `CO2`; aquí quedó una sola vez.

## 4. Diseño

### 4.1 Módulo nuevo `core/limpieza_datos.py` (sin dependencias de Django ni de Mongo)

Función única, pura y testeable:

```
limpiar_lecturas(df_crudo) -> ResultadoLimpieza(df_horario, reporte)
```

Pasos, en este orden:

1. **Normalizar nombres** de columnas (`fecha→datetime`, `temperatura→temp`, `humedad→humidity`, `presion→pressure`, `luz→light`), igual que hoy.
2. **Tipos:** `to_datetime(coerce)` y `to_numeric(coerce)`; filas sin fecha válida se descartan.
3. **Duplicados exactos** por (`idDispositivo`, `datetime`) y orden cronológico.
4. **Rango físico (P1, rango de validez de 3.1):** valores fuera de rango pasan a `NaN` (no se borra la fila). Los **picos plausibles no se tocan**: las vistas los presentan como eventos reales.
5. **Remuestreo por hora** (promedio) por dispositivo. Es lo que alinea el entrenamiento con el horizonte de predicción y neutraliza los repetidos de Redspira.
6. **Cobertura (P2):** una variable con menos cobertura que el mínimo se descarta y no se usa como característica. El objetivo (`PM2_5`/`CO2`) con cobertura insuficiente → no se entrena ese modelo.
7. **Selección de características por datos:** solo entre `temp`, `humidity`, `pressure`, `light` que hayan pasado el paso 6. Resuelve H1.

### 4.2 Reporte de limpieza

El resultado incluye un diccionario con: filas de entrada; filas descartadas por paso (fecha inválida, duplicados); valores anulados por rango, por variable; horas resultantes; variables descartadas con motivo; rango de fechas. Se devuelve en `ejecutar_analisis` como `reporte_limpieza`. **No cambia ninguna pantalla en este plan**: si se quiere mostrar, se define aparte con su mockup.

### 4.3 Puntos de integración

| Archivo | Cambio |
|---|---|
| `core/limpieza_datos.py` | Nuevo (4.1). |
| [run_engine.py](core/management/commands/run_engine.py) `ejecutar_analisis` | Reemplaza la normalización de las líneas 723-747 por `limpiar_lecturas`. `stats_ambientales` se calcula sobre el dato limpio. |
| `run_engine.py` `entrenar_modelo` | Recibe la lista de características elegida en el paso 7 en lugar de `base_features`. Se conservan `hour`, `dow` y el mínimo de P3. |
| `run_engine.py` `generar_features_futuras` / `predecir_futuro_simple` | Usan el DataFrame limpio y solo las características elegidas. |
| [run_model.py](core/management/commands/run_model.py) | Usa el mismo módulo; se elimina la limpieza duplicada (H5). |
| [views.py:873-1054](core/views.py#L873-L1054) | Sin cambios funcionales; recibe la clave nueva sin usarla. |
| [views.py:160-480](core/views.py#L160-L480) (análisis de datos) | **Fase B**, ver 4.5. |

### 4.4 Fuera de alcance (decisiones posteriores)

- Normalizar la **escritura** de las tres fuentes (documento común) y migrar el historial.
- Colección derivada con promedios horarios (solo si el rendimiento lo exige).
- **División temporal para las métricas.** Hoy se entrena y se evalúa con los mismos datos ([run_engine.py:90-91](core/management/commands/run_engine.py#L90-L91)), por lo que el R² sale inflado y el resumen ejecutivo lo traduce a "calidad alta". La limpieza no lo corrige. Recomendado como plan aparte.
- Mostrar el reporte de limpieza en pantalla.

### 4.5 Fase B: análisis de datos (KMeans, DBSCAN, gráficas)

Se hace después de la Fase A y con su propia aprobación, porque **cambia números que el usuario ve hoy en pantalla**.

- La vista llama a `limpiar_lecturas` por cada dispositivo seleccionado (los rangos P1 y la cobertura P2 son los mismos que en la Fase A). Las gráficas por dispositivo salen del resultado horario limpio.
- Para el clustering se conserva el comportamiento actual de promediar los dispositivos por hora, pero **solo con horas completas** en las variables usadas (P4). Se elimina `bfill().fillna(0)`.
- Los valores imposibles (fallas del sensor) desaparecen antes de escalar. Los **picos plausibles se conservan**, para que DBSCAN siga detectándolos como momentos atípicos reales.
- El reporte de calidad se alimenta del `reporte_limpieza`: filas inválidas, valores anulados por rango y horas sin dato. Se conservan las claves actuales (`total_rows_raw`, `rows_dropped`, `integrity_pct`, `invalid_per_col`, `imputed_per_col`) para no tocar la plantilla; cambia su significado según P5.
- Efecto visible esperado: los mínimos y máximos de las gráficas ya no incluirán valores imposibles, y la integridad pasará a reflejar nulos e inválidos. La tabla de datos puros y el CSV siguen mostrando el dato crudo.

## 5. Modelo de datos y despliegue

- **Sin migraciones** ni cambios en Mongo.
- A diferencia de la ingesta, este cambio **sí se despliega con el flujo normal**: la limpieza vive en el contenedor `web`, que el deploy reconstruye. No hay pasos manuales en producción.

## 6. Pruebas (`core/tests_limpieza_datos.py`, `SimpleTestCase`, datos sintéticos, sin Mongo)

1. Dispositivo tipo API (`presion` y `luz` en `None`): conserva las filas y descarta esas dos variables (regresión de H1).
2. Dispositivo tipo Redspira (sin llave `luz`): entrena con las variables disponibles.
3. Dispositivo tipo TTN con llaves nulas explícitas.
4. `CO2` como texto, float y entero mezclados: queda numérico.
5. Valores fuera de rango (`-999`, humedad 150): pasan a `NaN`; un pico plausible se conserva.
6. Duplicados exactos: se descartan.
7. Remuestreo: lecturas cada minuto y cada 10 minutos dan la misma rejilla horaria.
8. Cobertura por debajo del mínimo: la variable se descarta; objetivo sin cobertura: no se entrena.
9. Entrada vacía, sin columna de fecha, una sola fila: no truncan con error.
10. Contenido del reporte (conteos exactos en un caso armado a mano).
11. `ejecutar_analisis` con `core.views.get_mongo_data` simulado: un dispositivo de la API ahora entrena y devuelve métricas.

Fase B (vista de análisis de datos, con `get_mongo_data` simulado):

12. Con huecos al final de la serie, el clustering no recibe ceros falsos (regresión de H8).
13. Un `-999` de temperatura no aparece como anomalía de DBSCAN ni altera la escala; un pico plausible de PM2.5 sí se detecta.
14. El reporte de calidad cuenta nulos e inválidos (regresión de H9) y conserva las claves que usa la plantilla.
15. Con dos dispositivos seleccionados, las gráficas siguen mostrando una serie por dispositivo.

## 7. Verificación manual (local)

1. Cargar en el Mongo **local** (puerto 27018 del `docker-compose.yaml`) datos de ejemplo de las tres formas de documento. No se usa el Mongo de producción.
2. `python manage.py run_engine --device <id>` antes y después: comparar filas, variables usadas y si entrena.
3. Abrir la pantalla de análisis predictivo con un dispositivo tipo API y confirmar que ya aparecen métricas y predicción no plana.

## 8. Entrega

Fase local en la rama `Diego`, sin commit ni push hasta que el usuario apruebe la prueba. No se hace push a `main`. La implementación inicia cuando el usuario lo indique, confirme P1 (falta `luz`, y presión y CO2 por confirmar) y entregue P2.

## 9. Orden de implementación

1. Constantes con P1–P4 (con los valores de 3.1 ya confirmados por el usuario).
2. `core/limpieza_datos.py` y sus pruebas (1-10).
3. Integración en `ejecutar_analisis` y `entrenar_modelo`, y prueba 11.
4. Integración en `run_model.py`.
5. Verificación manual y aprobación de la Fase A.
6. **Fase B (4.5):** integrar en la vista de análisis de datos, pruebas 12-15 y verificación comparando la pantalla antes y después con los mismos dispositivos y fechas.

## 10. Observaciones secundarias (sin acción en este plan)

- [views.py:910](core/views.py#L910) llama `resultados.keys()` antes de comprobar `None`. Si el motor no devuelve resultados, falla con `AttributeError` y el aviso "No se pudieron generar resultados" nunca se muestra.
- Si `API_KEY` no está definida en el entorno, el API de sensores acepta peticiones sin llave ([views.py:1063](core/views.py#L1063)).
