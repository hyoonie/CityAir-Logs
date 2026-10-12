# Plan: integrar nuevo registro de Escuelas y Enlaces

## 1. Contexto

> **Entrega:** primero se integra y prueba solo en local; la carga a producción se hace después de que el usuario apruebe las pruebas. La implementación inicia únicamente cuando el usuario lo indique (ver sección 13).

El módulo de escuelas actual (rama base `Diego`) captura una escuela con solo 5 datos (nombre, municipio, estado, nivel, CCT) y crea al Enlace con un usuario y contraseña tecleados a mano. Los nuevos mockups piden un registro formal en **dos pasos**:

1. **Registrar escuela**: identificación, ubicación (con mapa) y convenio firmado en PDF.
2. **Registrar enlace**: responsable único de la escuela, con acceso mediante correo y contraseña temporal.

Objetivo: reemplazar ambos formularios y su modelo de datos manteniendo el resto del módulo (listado, detalle, sensores, gestión de docentes y alumnos) funcionando.

## 2. Decisiones confirmadas con el usuario

| Tema | Decisión |
|---|---|
| Estado de la escuela | Solo `activa` (sí/no) como hoy. Se ignoran "Solicitada / En validación" de los mockups. |
| Flujo | Dos pantallas: primero la escuela, luego el enlace (lista solo escuelas sin enlace). |
| Aviso de privacidad | **Fuera** (redundante con el PDF). |
| Vigencia del convenio | **Fuera**. |
| Tipo de documento | Select con una sola opción: `Convenio`. Modelado como campo para ampliarlo después. |
| "Elegir en el mapa" | **Dentro**, con Leaflet (ya se usa en `mapa-de-dispositivos.html`). |
| Niveles | Solo `secundaria`, `media_superior`, `superior`. Migración: `preparatoria`→`media_superior`, `universidad`→`superior`, `primaria`→`secundaria` (se encontró una primaria en la BD local; este último paso no tiene reverso). |
| CCT | Formato México (ver 4.1). Única por (país, CCT). Opcional. |
| Teléfono | Acepta `228 165 0525` o `228-165-0525`; se guarda `###-###-####`. |
| Correo | Lista blanca en constante: `gmail.com`, `edu.mx` y subdominios `*.edu.mx`. Aplica a correo institucional y a correo del enlace. |
| País | Select: México, Colombia (constante ampliable). |
| Obligatorios (escuela) | Los del mockup: nombre, nivel, sostenimiento, país, estado, municipio, latitud, longitud, tipo de documento, archivo PDF y fecha de firma. Opcionales (se validan solo si se capturan): CCT, turno, teléfono, correo, colonia, calle, número, código postal. |
| Abreviaturas en el nombre oficial | **No se valida.** Se evaluó una regex para rechazar siglas (p. ej. `ITSX` en lugar de "Instituto Tecnológico Superior de Xalapa") y se descartó por su complejidad técnica (hay nombres oficiales con siglas como CBTIS o CONALEP). El nombre solo se valida como obligatorio, con su longitud máxima (128). |
| Fecha de registro | Automática al guardar (no se captura). |
| Fecha de firma | No puede ser posterior a la fecha actual del sistema. |
| Acceso del Enlace | Correo + contraseña temporal. Se genera aleatoria, **se muestra una sola vez** al administrador y el Enlace **debe cambiarla en su primer ingreso**. |
| Invitación por correo | **Pendiente** (no hay SMTP). Se muestra la opción deshabilitada ("Próximamente"). |
| Usuario del Enlace | Se genera en automático: `<primer_nombre_normalizado>_<id_escuela>` (ej. `maria_12`). Si choca, se agrega sufijo numérico. Máximo 45 caracteres. |
| Apellidos | Se dividen: primera palabra → `aPaterno`, el resto → `aMaterno` (vacío si es un solo apellido). |
| Enlace único | Una escuela solo puede tener un Enlace. |
| Convenio PDF | Almacenamiento privado; lo ve/descarga el admin y el Enlace de esa escuela. |
| Alcance extra | Editar escuela y reemplazar PDF; editar enlace con campos nuevos; detalle muestra datos nuevos y descarga del convenio; la asignación de sensores se quita del registro pero se conserva en el detalle. |
| Permisos | Se mantienen: escuela = `is_staff` o "Administrador de sistema"; enlaces = solo superusuario. |

Pendiente menor, a validar al implementar: los mockups muestran los datos del enlace sin asterisco. Se tratan los cinco como obligatorios (nombre, apellidos, correo, teléfono, cargo), porque tu texto los lista como datos del enlace.

## 3. Cambios al modelo de datos ([core/models.py](core/models.py))

### 3.1 `Escuela` (campos nuevos o modificados)
- `nombre` (existente): nombre oficial.
- `clave_centro_trabajo` (existente): se quita `unique=True` global. Se agrega `UniqueConstraint(fields=['pais','clave_centro_trabajo'], condition=Q(clave_centro_trabajo__isnull=False))`.
- `nivel`: nuevas choices (`secundaria`, `media_superior`, `superior`).
- `sostenimiento`: `publico` / `privado`.
- `turno`: `matutino` / `vespertino` / `mixto`.
- `telefono` (CharField 12), `correo_institucional` (EmailField).
- `fecha_registro`: `DateTimeField(default=timezone.now, editable=False)`.
- Ubicación: `pais`, `colonia`, `calle`, `numero`, `codigo_postal`, `latitud` y `longitud` (`DecimalField(9,6)`). `estado` y `municipio` ya existen.
- Todos los campos nuevos son `null=True/blank=True` **a nivel BD** para no romper escuelas ya registradas; la obligatoriedad se exige en el formulario.
- `activa`: sin cambios.

### 3.2 Modelo nuevo `DocumentoEscuela`
- `escuela` (FK, `related_name='documentos'`), `tipo` (choices: `convenio`), `archivo` (FileField con storage privado), `fecha_firma` (DateField), `fecha_carga` (auto), `subido_por` (FK a `Usuario`, null).
- Un convenio vigente por escuela; "Reemplazar" sustituye el registro y borra el archivo anterior.

### 3.3 `Usuario` (campos nuevos)
- `telefono` (CharField 12, null), `cargo` (CharField 100, null), `debe_cambiar_password` (Boolean, default False).
- La regla "un solo Enlace por escuela" no se puede expresar como constraint (el tipo está en otra tabla). Se valida en el formulario y la vista, dentro de `transaction.atomic()` con `select_for_update()` sobre la escuela.

### 3.4 Migraciones
- `0006_escuela_datos_ampliados`: campos nuevos, `DocumentoEscuela`, `Usuario.*`, constraint.
- `0007_migrar_niveles_escuela`: `RunPython` con el mapeo de niveles (reversible).
- Hay que **versionarlas**: el deploy ejecuta `makemigrations core` en producción y generaría migraciones divergentes si no existen.

## 4. Validaciones y utilidades (archivo nuevo [core/validators.py](core/validators.py))

### 4.1 CCT México
Regex propuesta: `^(0[1-9]|[12]\d|3[0-2])[A-Z]{3}\d{4}[A-Z]$` (10 caracteres, se normaliza a mayúsculas).
- Posiciones 1-2: entidad federativa (INEGI).
- Posición 3: sostenimiento (por ejemplo `E`, `P`, `D`).
- Posiciones 4-5: tipo de servicio educativo (por ejemplo `PR`).
- Posiciones 6-9: folio consecutivo.
- Posición 10: dígito verificador (letra).
- El ejemplo del mockup, `30EPR0001X`, cumple. Solo se valida si `pais == México`. Fuente de la estructura: [CCT (N+)](https://www.nmas.com.mx/nacional/como-buscar-cct-escuela-2025-asi-puedes-sacar-clave-siged-sep-registro/) y [Mextudia](https://mextudia.com/clave-del-centro-de-trabajo-cct/). Se afinará con ejemplos reales antes de cerrar.

### 4.2 Otras
- `normalizar_telefono()`: acepta espacios o guiones, exige 10 dígitos y devuelve `###-###-####`.
- `validar_correo_dominio()`: lista blanca `DOMINIOS_PERMITIDOS = ('gmail.com', 'edu.mx')` con soporte de subdominios `*.edu.mx`.
- `validar_fecha_no_futura()`: `fecha_firma <= timezone.localdate()`.
- `validar_pdf()`: extensión `.pdf`, cabecera `%PDF-` y tamaño máximo (propuesta: 5 MB, constante).
- `generar_password_temporal()`: `get_random_string` (12 caracteres) o `secrets`.
- `generar_usuario_enlace(nombre, escuela_id)`: slug sin acentos, truncado a 45, con control de colisiones.

## 5. Formularios ([core/forms.py](core/forms.py))
Hoy solo contiene `UploadFileForm`; se agrega:
- `EscuelaForm` (ModelForm) con choices, validaciones de 4 y normalización.
- `ConvenioForm` (`tipo`, `archivo`, `fecha_firma`). En edición el archivo es opcional ("Reemplazar").
- `EnlaceForm` (`nombre`, `apellidos`, `correo`, `telefono`, `cargo`, `escuela`).
  - `escuela` limitada a escuelas sin Enlace.
  - Correo único contra `Usuario.email`.
  - Correo ≤ 255 caracteres.

## 6. Vistas y URLs ([core/views.py](core/views.py), [core/urls.py](core/urls.py))

| Acción | Cambio |
|---|---|
| `view_escuelas_registrar` (≈1292) | Reescribir con `EscuelaForm` + `ConvenioForm` en una transacción. Quitar la selección de sensores del registro. Mismos permisos. Al guardar, redirige a "Registrar enlace" con la escuela preseleccionada. |
| `view_escuelas_editar` (nueva, `escuelas/<id>/editar/`) | Misma pantalla en modo edición, con "Reemplazar" PDF. Mismos permisos que registrar. |
| `view_escuelas_crear_enlace` (≈1203) | Reemplazar por el flujo del mockup 2. Lista solo escuelas sin Enlace. Tarjeta de resumen de la escuela elegida (datos vía `json_script` + JS). Crea `Usuario` (`tipousuario=4`, usuario autogenerado, `debe_cambiar_password=True`, contraseña temporal). Se conserva el nombre de URL `escuelas-crear-enlace` para no romper enlaces ni tests. |
| Mostrar credenciales | Tras crear, guarda correo + contraseña temporal en `request.session` y los muestra **una sola vez** en una pantalla de confirmación (no en `messages`). Se limpian al leerlas. |
| `view_escuelas_editar_enlace` (≈1242) | Agregar apellidos divididos, teléfono, cargo y correo. Opción "Restablecer contraseña temporal" que repite el flujo de mostrar una vez y fuerza cambio. |
| `view_escuelas_detalle` (≈1324) | Mostrar identificación, ubicación y convenio (descarga). Se conserva la asignación de sensores. |
| `view_escuelas_convenio` (nueva, `escuelas/<id>/convenio/`) | `FileResponse` del PDF. Permiso: superusuario/staff/"Administrador de sistema", o el Enlace de esa escuela. Resto: 403. |
| `view_cambiar_password` (≈85) | Al cambiar con éxito, poner `debe_cambiar_password=False`. |
| `view_ct_login` (≈54) | Tras login, si `debe_cambiar_password`, redirigir a cambio de contraseña. |

## 7. Almacenamiento privado del PDF
- Storage propio en `core/storage.py` (`FileSystemStorage`) con ubicación tomada de la variable de entorno `CONVENIOS_ROOT` (por defecto `BASE_DIR/private_media/convenios`). Fuera de `static/`, nunca servido directo.
- **Por qué no `MEDIA_ROOT`:** `*/settings.py` está en `.gitignore`, así que cualquier cambio a `settings.py` no viaja con el repo. Con este enfoque el código funciona sin tocarlo.
- Infraestructura fuera del repo (`docker-compose.yaml`, `Dockerfile` y `/nginx` también están ignorados): agregar un volumen persistente para esa ruta en servidor y local, o el PDF se perderá al recrear el contenedor `web`. **Paso manual de despliegue.**

## 8. Forzar cambio de contraseña (middleware)
- `core/middleware.py`: si el usuario está autenticado y `debe_cambiar_password`, redirige a la pantalla de cambio, salvo logout, cambio de contraseña y archivos estáticos.
- Requiere registrar el middleware en `MIDDLEWARE` de `settings.py`, que no está versionado. **Paso manual en local y servidor.**
- Sin ese paso el cambio sigue forzándose al iniciar sesión (redirección en `view_ct_login`), pero no se bloquea la navegación directa por URL.

## 9. Plantillas ([core/templates/escuelas/](core/templates/escuelas/))
- `registrar-escuela.html`: reescribir con secciones **Identificación / Ubicación / Autorización** del mockup 1, indicador `*`, botones Cancelar / Guardar escuela y errores por campo.
- Mapa: modal o panel Leaflet (cargado igual que en el mapa de dispositivos) que rellena latitud y longitud al hacer clic; los campos manuales siguen disponibles.
- `crear-enlace.html` (título "Registrar enlace"): mockup 2 con selector de escuelas sin Enlace, tarjeta de resumen, sección **Datos del enlace** y sección **Acceso**. La opción "Enviar invitación por correo" va deshabilitada con la leyenda "Próximamente"; "Generar contraseña temporal" va seleccionada.
- Nueva plantilla de confirmación de credenciales, que se muestra una sola vez.
- Actualizar `editar-enlace.html`, `detalle-escuela.html` y `escuelas.html` (nivel, nuevos datos, ícono de convenio). Nueva `editar-escuela.html` o reutilizar la de registro en modo edición.
- El estilo de nav y colores de los mockups se resuelve con la `base.html` existente; no se rediseña la navegación.

## 10. Pruebas ([core/tests.py](core/tests.py))
- Actualizar `RegistrarEscuelaTest` y las pruebas de crear/editar enlace, porque hoy envían los campos antiguos.
- Casos nuevos:
  - Validadores: CCT válido e inválido, teléfono con espacios y con guiones, dominios permitidos y rechazados, fecha de firma futura.
  - PDF no válido o demasiado grande.
  - CCT duplicado en el mismo país vs distinto país.
  - Segundo enlace para la misma escuela rechazado.
  - Contraseña temporal mostrada una sola vez y forzado de cambio.
  - Descarga de convenio: admin y Enlace propio sí; Enlace de otra escuela y docente no.
  - Migración de niveles (`preparatoria`→`media_superior`, `universidad`→`superior`).

## 11. Verificación
1. `python manage.py makemigrations core --check --dry-run`, luego `python manage.py migrate` en BD local con una escuela de ejemplo (`preparatoria`/`universidad`) previa a la migración.
2. `python manage.py test core`.
3. Prueba manual con `runserver`:
   - Registrar escuela con el ejemplo del mockup (`30EPR0001X`, 228 165 0525, convenio PDF).
   - Registrar su enlace y copiar la contraseña temporal.
   - Iniciar sesión con correo + contraseña temporal y comprobar el cambio forzado.
   - Descargar el convenio como Enlace propio y como otro usuario (debe dar 403).
4. Comprobar que `escuelas/`, detalle, gestión de usuarios y asignación de sensores siguen funcionando.

## 12. Observaciones secundarias (sin acción ahora)
- `docker-compose.yaml` y credenciales viven fuera de git; no se tocan.
- La invitación por correo queda como trabajo futuro (requiere SMTP y variables `EMAIL_*`).
- Apellidos compuestos ("de la Cruz") se dividirán por la primera palabra; si esto resulta incómodo se puede cambiar a un solo campo `apellidos`.

## 13. Estrategia de entrega (indicada por el usuario)
- **Fase local primero:** la integración se desarrolla y prueba únicamente en local (`runserver` / docker-compose local). No se toca producción, ni se hace push a `main` (que dispara el deploy automático).
- **Producción después:** solo cuando el usuario apruebe las pruebas en local se prepara la carga a producción (PR, volumen de convenios, middleware en `settings.py` del servidor, migraciones).
- **Inicio de la implementación:** no se empieza a codificar hasta que el usuario dé la instrucción explícita de iniciar. Este documento solo define el plan.

## 14. Orden de implementación sugerido (fase local)
1. Modelos y migraciones 0006 y 0007.
2. `validators.py`, `storage.py` y formularios.
3. Vistas de escuela (registrar, editar, convenio, detalle) con sus plantillas.
4. Vistas de enlace (registrar, credenciales una vez, editar) y forzado de cambio de contraseña.
5. Pruebas y verificación local; el usuario valida.
6. (Después de aprobación) Pasos manuales de infraestructura en producción: volumen de convenios y middleware en `settings.py`.

## 15. Anexo: protocolo de aviso por mockup
- Los mockups a integrar son dos: **Registrar escuela** y **Registrar enlace**.
- Cuando termine de integrar **y probar** un mockup, debo avisar con esta frase exacta:
  - "Listo patroncito he integrado + *nombre del mockup*", por ejemplo "Listo patroncito he integrado Registrar escuela".
- El aviso se da por cada mockup, al terminar su integración y sus pruebas locales, sin esperar a tener ambos listos.
- No se da el aviso si las pruebas del mockup fallan o quedan pendientes. En ese caso se reporta el estado real.

## 16. Pendiente a corto plazo: catálogos cerrados (Estado, Municipio y Cargo)
Registrado por el usuario el 2026-10-07 y definido el 2026-10-08. **Análisis y propuesta; no implementado todavía.**

### 16.1 Definición (confirmada)
Estandarizar = catálogo cerrado, para evitar problemas de calidad de la información.
- **Escuela:** `estado` y `municipio` se eligen de un catálogo cerrado de México (selects dependientes: primero el estado, luego sus municipios).
- **Enlace:** `cargo` se elige de un catálogo cerrado: **Director(a)**, **Subdirector(a)** y **Coordinador(a)**.
- El catálogo de estados y municipios debe ser **persistente** (guardado en la BD del proyecto) y **revisarse contra una API cada semana o mes** para detectar actualizaciones.

### 16.2 ¿El proyecto puede construirlo hoy?
**No.** Hallazgos del análisis del código y de la BD local:
- No existe ningún catálogo, fixture ni librería de estados y municipios. `estado` y `municipio` de `Escuela` son `CharField` libres, sin validación.
- `geopy`/Nominatim solo geocodifica direcciones; no sirve como catálogo.
- No hay planificador de tareas (ni Celery, ni cron, ni APScheduler). Lo más cercano son los comandos de larga duración de `management/commands` (`mqtt_listener`, `redspira_worker`), cuya ejecución vive fuera del repo.
- Los datos actuales **no coinciden con los nombres oficiales**: las escuelas locales tienen `Veracruz` (oficial: *Veracruz de Ignacio de la Llave*) y `Xalapa-Enriquez` (oficial: *Xalapa*). Hace falta un mapeo para los registros existentes.
- `Escuela.municipio` tiene `max_length=64`, pero el nombre oficial más largo del catálogo mide más de 70 caracteres (*Heroica Villa Tezoatlán de Segura y Luna, Cuna de la Independencia de Oaxaca*). Hay que ampliarlo (propuesta: 150).
- Cargos existentes en la BD local (texto libre): `Director General`, `Directora General`, `Sudirector`.

### 16.3 Fuente de datos recomendada: INEGI
Catálogo Único de Claves de Áreas Geoestadísticas Estatales, Municipales y Localidades (AGEEML). Fuente oficial de las claves de entidad y municipio de México.
- Servicio web con JSON, **sin token ni registro** (verificado el 2026-10-08): documentación en <https://www.inegi.org.mx/servicios/catalogoUnico.html>.
  - Estados: `https://gaia.inegi.org.mx/wscatgeo/v2/mgee/` → 32 estados (`cve_ent`, `nomgeo`, `nom_abrev`).
  - Municipios por estado: `https://gaia.inegi.org.mx/wscatgeo/v2/mgem/{cve_ent}` → p. ej. `mgem/30` devuelve 212 municipios de Veracruz (`cvegeo`, `cve_ent`, `cve_mun`, `nomgeo`).
- Medido: **32 estados y 2,478 municipios** en 33 peticiones (~22 s en total).
- INEGI publica el catálogo **mensualmente** (según su calendario de difusión). Por eso una revisión semanal o mensual es suficiente.
- Descarga alternativa en bloque (archivos ZIP/CSV): <https://www.inegi.org.mx/app/ageeml/>. Sirve de respaldo si el servicio web cambia.
- A confirmar: los términos de uso de INEGI para publicar los datos (se espera uso libre citando la fuente).

### 16.4 Propuesta de diseño
**Modelos nuevos**
- `Estado`: `cve_ent` (único, 2 dígitos), `nombre`, `nombre_abreviado`, `activo`.
- `Municipio`: `cvegeo` (único, 5 dígitos), `estado` (FK, `PROTECT`), `cve_mun`, `nombre` (hasta 150), `activo`.
- `SincronizacionCatalogo`: bitácora (fecha, resultado, altas, cambios de nombre, bajas, error).
- `Escuela`: se agregan FK nulas a `Estado` y `Municipio`. Se conservan las columnas de texto `estado` y `municipio` como copia del nombre, para no romper el listado, el detalle ni los reportes. Se rellenan solas al guardar y cuando el catálogo renombra algo.
- Los registros **no se borran nunca**: si INEGI retira un municipio, queda `activo=False` y las escuelas que lo usan no se pierden.

**Persistencia inicial (sin depender de la API en el despliegue)**
- Un archivo `core/data/catalogo_geo_mx.json` versionado en el repo, cargado por una migración de datos. El despliegue ya corre `migrate`, así que el catálogo queda poblado en local y en producción sin llamar a la API.

**Actualización periódica**
- Comando nuevo `python manage.py sincronizar_catalogo_geo` (con `--dry-run`). Consulta a INEGI y hace `upsert` por clave (`cve_ent`, `cvegeo`):
  - Detecta altas, cambios de nombre y bajas.
  - **No aplica nada si la respuesta es sospechosa** (p. ej. menos de 32 estados o menos de ~2,400 municipios, o error de red). Todo se hace en una transacción.
  - Registra el resultado en `SincronizacionCatalogo`.
- Programación: como el proyecto no tiene planificador, se propone una línea de cron en el servidor (semanal) que ejecute `docker compose exec -T web python manage.py sincronizar_catalogo_geo`. Alternativa: un workflow de GitHub Actions con `schedule` que lo ejecute por SSH, reutilizando los secretos del deploy.
- Es un **paso manual de infraestructura** (el cron vive fuera del repo), igual que el volumen de convenios y el middleware.

**Formularios**
- `EscuelaForm`: selects de Estado y Municipio; el municipio se carga según el estado elegido mediante un endpoint JSON interno que lee **de la BD local** (`/api/catalogo/estados/<id>/municipios/`, con login). Nunca se consulta INEGI durante una petición del usuario.
- `EnlaceForm` y `Usuario.cargo`: select con `Director(a)`, `Subdirector(a)` y `Coordinador(a)`; se guarda como código (`director`, `subdirector`, `coordinador`).
- Pruebas: catálogo cargado, estado y municipio coherentes entre sí, rechazo de combinaciones inválidas, sincronización con una respuesta simulada (altas, renombres, bajas, respuesta incompleta) y cargos.

**Migración de datos existentes**
- Escuelas: se empareja por nombre sin acentos ni mayúsculas, con una tabla de alias para los nombres cortos de uso común (`Veracruz`, `Coahuila`, `Michoacán`, `Xalapa-Enriquez`…). Lo que no se pueda emparejar queda sin FK y se lista para corrección manual; no se pierde el texto original.
- Cargos: se emparejan por prefijo (`Director…` → director, `Subdirector…`/`Sudirector` → subdirector, `Coordinador…` → coordinador). El resto queda vacío y se pide capturar de nuevo al editar el enlace.

### 16.5 Alcance fuera de esta sección
- `Dispositivo.estado` y `Dispositivo.ciudad` también son texto libre; no forman parte de este pendiente.
- Otros países (Colombia aparece en el select de país): pendiente de decidir cómo se capturan estado y municipio (ver preguntas abiertas).

### 16.6 Decisiones confirmadas (2026-10-08)
- **Otros países:** México usa el catálogo cerrado; Colombia (u otro) mantiene estado y municipio como texto libre.
- **Revisión periódica:** cron en el servidor, **semanal**.
- **Escuelas existentes:** emparejado automático con alias y revisión manual de lo que no coincida.

### 16.7 Estado de la implementación (local, 2026-10-08)
**Implementado y probado en local; sin commit ni producción.**
- Modelos `Estado`, `Municipio` y `SincronizacionCatalogo`; `Escuela.estado_catalogo` y `Escuela.municipio_catalogo` (FK `PROTECT`); `Escuela.municipio` ampliado a 150; `Usuario.cargo` con opciones cerradas (`director`, `subdirector`, `coordinador`).
- Migraciones: `0008_catalogo_geo_y_cargos` (esquema) y `0009_cargar_catalogo_geo` (carga el catálogo desde `core/data/catalogo_geo_mx.json`, empareja escuelas y mapea cargos). Se aplicaron en la BD local: 32 estados, 2,478 municipios, las 4 escuelas locales emparejadas (`Veracruz` → *Veracruz de Ignacio de la Llave*, `Xalapa-Enriquez` → *Xalapa*) y los 3 cargos existentes mapeados.
- Servicio `core/catalogo_geo.py` y comando `python manage.py sincronizar_catalogo_geo [--dry-run] [--archivo RUTA]`. No aplica respuestas incompletas (mínimo 32 estados y 2,400 municipios), no borra nada, reintenta 3 veces y registra cada ejecución en `SincronizacionCatalogo` (visible en el admin). Sale con error si falla, para que el cron lo detecte.
- Endpoint interno `GET /api/catalogo/estados/<id>/municipios/` (con login, lee de la BD local).
- Formulario de escuela: selects dependientes Estado → Municipio para México; texto libre para otros países. Formulario de enlace: select de cargo.
- Pruebas nuevas en `core/tests_catalogo.py`.

**Paso manual en producción (cron semanal)**, por ejemplo los lunes a las 03:00:
```
0 3 * * 1  cd /root/CityAirLogs_Infra/CityAir-Logs && docker compose exec -T web python manage.py sincronizar_catalogo_geo >> /var/log/catalogo_geo.log 2>&1
```
Notas para producción:
- Las migraciones 0008 y 0009 deben subirse con el código (el despliegue ya ejecuta `migrate`). No requieren llamar a la API.
- Hay que subir también `core/data/catalogo_geo_mx.json` (lo lee la migración 0009).
- Los cargos que no se puedan mapear a un código quedan vacíos y se piden de nuevo al editar el enlace.
