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

## 16. Pendiente a corto plazo: estandarización de campos
Registrado por el usuario el 2026-10-07. **No se implementa todavía**; se integrará a corto plazo, después de aprobar las pruebas de los dos mockups.

Campos que faltan por estandarizar:
- **Escuelas:** `estado` y `municipio`.
- **Enlace:** `cargo`.

Hoy los tres son texto libre. Por definir antes de implementar (no se asume nada):
- Qué significa "estandarizar" en cada caso: catálogo cerrado (select), autocompletado con valores sugeridos, o normalización del texto (mayúsculas, acentos, espacios).
- Fuente del catálogo: lista de estados y municipios (¿solo México, también Colombia?) y lista de cargos permitidos.
- Qué hacer con los registros ya existentes cuyo texto no coincida con el catálogo.
