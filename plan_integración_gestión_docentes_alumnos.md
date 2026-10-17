# Plan: gestión de docentes y alumnos por el Enlace

## 1. Contexto

Hoy el Enlace entra a `escuelas/gestionar-usuarios/` ([core/views.py](core/views.py), `view_escuelas_gestionar_usuarios`, ≈1547) y puede dar de alta un usuario a la vez (Docente o Alumno) tecleando a mano usuario, correo, nombre, apellido paterno y contraseña. La lista de usuarios de la escuela no tiene búsqueda, filtros ni paginación.

Este plan agrega cuatro actividades:

| # | Actividad | Criterio de aceptación |
|---|---|---|
| A | Endpoint y vista para que el Enlace registre **un** docente, con validación de rol y `escuela_id`. | El docente aparece en el padrón de la escuela correcta y recibe acceso de solo consulta según su rol. |
| B | Formulario en el frontend con carga de CSV para registrar alumnos en lote. **Se conserva el registro individual de alumnos.** | Permite agregar filas dinámicamente, valida campos en línea y muestra errores sin recargar la página. |
| C | Endpoint que recibe un arreglo de alumnos y los crea en una sola transacción, validando duplicados, formato de correo y pertenencia a la escuela del Enlace. | 50 alumnos en menos de 5 s, sin registros huérfanos ni duplicados. |
| D | Vista que lista docentes y alumnos de la escuela del Enlace, con búsqueda por nombre/correo y filtro por rol. | El Enlace ve solo usuarios de su escuela; busca por nombre y filtra por rol; la paginación funciona con 100+ registros. |

## 2. Hallazgos del código actual que condicionan el diseño

1. **Hash de contraseña (riesgo para el criterio de 5 s).** Medí `make_password` en el entorno local (Django 5.2.7, PBKDF2 con 1,000,000 iteraciones): **0.68 s por usuario**. Crear 50 alumnos con el hash normal tardaría ~34 s. Se resuelve con el punto 4.4.
2. **El alta actual no valida casi nada.** No revisa dominio de correo, duplicados con mensaje claro ni formato de teléfono; los errores se muestran como la excepción cruda (`Error al crear usuario: ...`).
3. **El alta usa `escuela = request.user.escuela` solo si el tipo es Enlace.** Si entra un `is_staff` sin tipo Enlace, `escuela` queda en `None` y crearía usuarios huérfanos (sin escuela). El nuevo diseño exige un Enlace con escuela asignada.
4. **Activar/desactivar** filtra por `escuela=escuela` pero no por rol: con un `user_id` propio el Enlace podría desactivarse a sí mismo. Como esa vista se reescribe, se restringe a Docente y Alumno.
5. `Usuario.email` y `Usuario.usuario` son únicos **globales**. Un correo registrado en otra escuela cuenta como duplicado.
6. **Los docentes y alumnos ya son de solo consulta**: no existe ninguna vista de escritura para esos roles; `view_escuelas` les muestra la lectura de su sensor, la comparativa y (al Docente) la lista de alumnos; el CSV de `dato-puro` solo lo descargan superusuario y Enlace. Lo que falta es **demostrarlo con pruebas** (sección 9).
7. Ya existen y se reutilizan: `Usuario.telefono`, `Usuario.debe_cambiar_password`, el middleware `ForzarCambioPasswordMiddleware`, `normalizar_telefono`, `validar_correo_dominio`, `dividir_apellidos`, `generar_usuario_enlace`, `generar_password_temporal` ([core/validators.py](core/validators.py)) y los ids de rol 5 (Docente) y 6 (Alumno).

## 3. Decisiones confirmadas

| Tema | Decisión |
|---|---|
| Acceso | Contraseña temporal autogenerada (12 caracteres) + cambio obligatorio en el primer ingreso, igual que el Enlace. |
| Correo | Misma lista blanca que el Enlace: `gmail.com`, `edu.mx` y `*.edu.mx`. |
| Registro individual de alumnos | Se conserva, alineado al estándar nuevo: usuario autogenerado, contraseña temporal, correo validado y escuela fijada a la del Enlace. |

## 4. Diseño

### 4.1 Reglas comunes (docente, alumno individual y lote)
- **Quién puede:** solo un usuario con `tipousuario_id == 4` (Enlace), con `escuela_id` no nulo y escuela `activa`. Cualquier otro rol recibe 403. Un superusuario o administrador no registra docentes/alumnos por estas rutas (no tiene escuela de contexto).
- **`escuela_id` nunca se toma del cliente.** Siempre se usa `request.user.escuela_id`. Si la petición trae un `escuela_id` distinto, se responde 403 y no se guarda nada.
- **Rol asignado por el servidor:** 5 para el endpoint de docente, 6 para los de alumnos. El cliente no elige el rol.
- **Campos por persona:**

| Campo | Obligatorio | Regla |
|---|---|---|
| `nombre` | Sí | ≤ 45 caracteres, espacios normalizados |
| `apellidos` | Sí | Se dividen con `dividir_apellidos` (paterno ≤ 45, materno ≤ 45) |
| `correo` | Sí | Formato válido, ≤ 255, minúsculas, dominio de la lista blanca, único (sin distinguir mayúsculas) |
| `telefono` | No | Si se captura, `normalizar_telefono` (`###-###-####`) |

- **Usuario (login) autogenerado:** `<primer_nombre>_<id_escuela>`, con sufijo numérico si choca (reusa `generar_usuario_enlace`). El login por correo ya funciona (`UsuarioOrEmailBackend`), así que el usuario casi no se teclea.
- **Contraseña temporal:** `generar_password_temporal()`, `debe_cambiar_password=True`, `is_active=True`.

### 4.2 Módulo de servicio nuevo: `core/usuarios_escuela.py`
Concentra la lógica para que las tres vías (docente, alumno individual y lote) usen exactamente las mismas reglas:
- `validar_persona(datos)`: valida y normaliza una fila **sin tocar la BD** (campos, formato y dominio de correo, teléfono). Devuelve datos limpios o la lista de errores por campo.
- `validar_filas(filas)`: aplica `validar_persona` a cada fila y agrega las comprobaciones de conjunto, con **una sola consulta** a la BD:
  - correos repetidos dentro del mismo envío (se marcan las filas repetidas, no la primera);
  - correos ya existentes (`Lower(email) IN (...)`).
- `crear_usuarios(escuela, rol_id, filas_limpias)`: dentro de `transaction.atomic()`:
  1. precarga los usuarios ocupados con el prefijo de la escuela (una consulta) para generar logins sin consultar uno por uno;
  2. genera usuario y contraseña temporal por fila;
  3. inserta con **un solo `bulk_create`**;
  4. cualquier `IntegrityError` (carrera entre dos Enlaces con el mismo correo) revierte todo y devuelve un error de conflicto.
  Devuelve la lista `{fila, nombre, correo, usuario, password}`.
- Constantes: `ROL_DOCENTE = 5`, `ROL_ALUMNO = 6`, `MAX_FILAS_LOTE = 200`.
- Permiso: `enlace_de_escuela(user)` devuelve la escuela o `None`.

### 4.3 Endpoints y vistas

| Método y ruta | Nombre | Qué hace |
|---|---|---|
| `GET /escuelas/usuarios/` | `escuelas-gestionar-usuarios` (se **conserva el nombre**) | Actividad D: lista con búsqueda, filtro y paginación. Mantiene activar/desactivar. |
| `GET /escuelas/usuarios/registrar/` | `escuelas-registrar-usuarios` | Pantalla con tres pestañas: **Docente**, **Alumno individual**, **Alumnos en lote (CSV)**. |
| `POST /api/escuelas/docentes/` | `api-escuelas-docente-crear` | Actividad A: crea **un** docente. Recibe JSON `{nombre, apellidos, correo, telefono}`. |
| `POST /api/escuelas/alumnos/` | `api-escuelas-alumno-crear` | Alumno individual (misma lógica que el docente, rol 6). |
| `POST /api/escuelas/alumnos/validar/` | `api-escuelas-alumnos-validar` | Validación sin guardar de un arreglo de filas: devuelve errores por fila/campo (incluye duplicados contra la BD). Lo usa el frontend para validar en línea. |
| `POST /api/escuelas/alumnos/lote/` | `api-escuelas-alumnos-lote` | Actividad C: crea el arreglo completo en una transacción. |
| `GET /escuelas/usuarios/plantilla-alumnos.csv` | `escuelas-plantilla-alumnos` | Descarga la plantilla CSV (`nombre,apellidos,correo,telefono`). |

**Contrato del lote** (`POST /api/escuelas/alumnos/lote/`):
- Entrada: `{"filas": [{"nombre": "...", "apellidos": "...", "correo": "...", "telefono": ""}, ...]}` (`escuela_id` opcional; si viene y no coincide con la del Enlace → 403).
- `201`: `{"creados": 50, "usuarios": [{"fila": 1, "nombre": "...", "correo": "...", "usuario": "...", "password": "..."}, ...]}`.
- `400`: `{"errores": [{"fila": 3, "campo": "correo", "mensaje": "..."}]}` — **no se guarda ninguna fila** (todo o nada). También para arreglo vacío o con más de 200 filas.
- `403`: no es Enlace, no tiene escuela o `escuela_id` ajeno. `409`: conflicto de unicidad por carrera (se revierte todo).
- Todas las respuestas llevan `never_cache`. Exigen sesión y CSRF (`X-CSRFToken` desde el JS).
- Mensajes de error en español (mismo mecanismo `en_espanol` de las vistas de escuelas).

Las vistas HTML y los endpoints JSON son delgados: solo permisos, lectura de la petición y llamada al servicio.

### 4.4 Rendimiento: hash ligero para contraseñas temporales
- Se define en el código una subclase de `PBKDF2PasswordHasher` con `iterations = 10_000` y se usa **directamente** (`hasher.encode(password, hasher.salt())`), sin tocar `settings.py` (que no se versiona).
- Es seguro para este caso: la contraseña temporal es aleatoria de 12 caracteres (~69 bits de entropía) y es de un solo uso.
- **Verificado en local:** 50 hashes ligeros = **0.36 s** (contra ~34 s con el hash normal). El hash resultante (`pbkdf2_sha256$10000$...`) es válido para el login y, como las iteraciones difieren de las configuradas, **Django lo vuelve a hashear con el algoritmo normal en el primer ingreso exitoso**. Además el usuario debe cambiar la contraseña de inmediato.
- Presupuesto esperado para 50 alumnos: validación (1 consulta) + usuarios ocupados (1 consulta) + hash (~0.4 s) + `bulk_create` (1 inserción) ≈ **menos de 1 s**.

### 4.5 Entrega de credenciales (solo una vez)
- La respuesta `201` trae las contraseñas temporales; el frontend las muestra en un modal/tabla **una sola vez** y ofrece **“Descargar credenciales (CSV)”** generado en el navegador (Blob) con `nombre, correo, usuario, contraseña temporal`. No se guardan en sesión ni en la BD.
- El CSV exportado neutraliza celdas que empiecen con `=`, `+`, `-` o `@` (prefijo `'`) para evitar inyección de fórmulas en Excel.
- Si el Enlace pierde las credenciales, puede usar **“Restablecer contraseña temporal”** en la lista (acción nueva por usuario, solo Docente/Alumno de su escuela; vuelve a mostrar la nueva una sola vez). Es el único alcance extra, necesario para no dejar cuentas inaccesibles.

### 4.6 Frontend de carga en lote
Plantilla [core/templates/escuelas/registrar-usuarios.html](core/templates/escuelas/registrar-usuarios.html) y script nuevo `core/static/js/escuelas/carga-alumnos.js` (se publica con el `collectstatic` que ya corre el despliegue).
- **Filas dinámicas:** tabla con columnas nombre, apellidos, correo y teléfono; botones “Agregar fila” y “Eliminar fila”, contador de filas (máx. 200).
- **Carga de CSV:** un botón/zona para elegir `.csv`. Se lee **en el navegador** (`FileReader`, UTF-8 con o sin BOM), se detecta el separador `,` o `;` (Excel en español usa `;`), se reconocen los encabezados (`nombre`, `apellidos`, `correo`, `telefono`) sin importar mayúsculas/acentos y las filas se cargan en la misma tabla para poder corregirlas antes de enviar. Límite de archivo: 1 MB.
- **Validación en línea, sin recargar:**
  - en el cliente, al salir de cada campo: obligatorio, longitud, formato y dominio de correo (la lista blanca llega desde el servidor con `json_script`, no se duplica a mano), teléfono, y correos repetidos dentro de la tabla;
  - en el servidor, con el botón **“Validar”** y de forma automática antes de enviar: `api-escuelas-alumnos-validar` devuelve los duplicados contra la BD, que se pintan en la fila y campo correspondiente.
- **Envío:** `fetch` JSON a `api-escuelas-alumnos-lote`; botón deshabilitado mientras responde (evita doble envío). Si hay errores se marcan las filas y no se pierde lo capturado. Si se crean, se muestra el modal de credenciales y la tabla se limpia.
- **Pestañas Docente y Alumno individual:** formularios simples con la misma validación en línea, que llaman a `api-escuelas-docente-crear` y `api-escuelas-alumno-crear`.
- Accesibilidad básica: errores con `aria-invalid`/texto junto al campo, no solo color.

### 4.7 Lista de usuarios (actividad D)
En `view_escuelas_gestionar_usuarios` (misma URL lógica, el nombre de la ruta no cambia):
- **Alcance:** `Usuario.objects.filter(escuela_id=<escuela del Enlace>, tipousuario_id__in=[5, 6])`, excluyendo al propio Enlace. No acepta ningún parámetro que cambie la escuela.
- **Búsqueda `q`:** se divide en palabras; cada palabra debe aparecer en `nombre`, `aPaterno`, `aMaterno`, `email` o `usuario` (AND entre palabras, OR entre campos), así “maria lopez” encuentra a María López.
- **Filtros:** `rol` (`todos`, `docente`, `alumno`) y `estado` (`todos`, `activos`, `inactivos`).
- **Paginación:** `Paginator` de 25 por página, orden estable (`tipousuario_id, aPaterno, nombre, pk`), los parámetros `q`, `rol` y `estado` se conservan al cambiar de página. Fuera de rango → última página.
- `select_related('tipousuario')`; contador “mostrando X de Y”.
- Acciones por fila: activar/desactivar (restringido a Docente/Alumno de su escuela) y restablecer contraseña temporal.
- Botón visible “Registrar usuarios” hacia la pantalla de la sección 4.3.

## 5. Modelo de datos y migraciones
No se esperan cambios de esquema: `Usuario` ya tiene `telefono`, `debe_cambiar_password`, `escuela` y `tipousuario`, y `email` y `usuario` ya son únicos. Se verificará con `makemigrations core --check --dry-run` (el despliegue corre `makemigrations`, así que no debe generar nada).

## 6. Archivos a crear o modificar
- Nuevos: `core/usuarios_escuela.py`, `core/templates/escuelas/registrar-usuarios.html`, `core/static/js/escuelas/carga-alumnos.js`, `core/tests_usuarios_escuela.py`.
- Modificados: [core/views.py](core/views.py) (vista de lista reescrita, vistas/endpoints nuevos), [core/urls.py](core/urls.py), [core/templates/escuelas/gestionar-usuarios.html](core/templates/escuelas/gestionar-usuarios.html), [core/templates/escuelas/escuelas.html](core/templates/escuelas/escuelas.html) y [detalle-escuela.html](core/templates/escuelas/detalle-escuela.html) solo si el texto del botón cambia.
- [core/tests.py](core/tests.py): `GestionarUsuariosTest.test_enlace_puede_crear_alumno` envía los campos antiguos (usuario y contraseña manuales); se actualiza al flujo nuevo.

## 7. Seguridad
- Sesión + CSRF en todos los POST; `never_cache` en las respuestas con contraseñas.
- Permiso y escuela se resuelven en el servidor en cada petición.
- Consultas con el ORM (sin SQL armado a mano); la búsqueda usa parámetros.
- Salida de errores en JSON sin trazas internas.
- Los mensajes de duplicado no revelan a qué escuela pertenece el correo ya existente.

## 8. Pruebas ([core/tests_usuarios_escuela.py](core/tests_usuarios_escuela.py))
- **Permisos:** Docente, Alumno, usuario sin escuela, Enlace sin escuela asignada, superusuario y anónimo → 403 o redirección a login en todos los endpoints.
- **`escuela_id` ajeno:** un Enlace que manda el `escuela_id` de otra escuela recibe 403 y no se crea nada; con `escuela_id` propio funciona.
- **Docente (A):** se crea con rol 5 y la escuela del Enlace; aparece en la lista; inicia sesión con correo y contraseña temporal, debe cambiarla, y al hacerlo accede a `escuelas/`.
- **Solo consulta:** un Docente y un Alumno reciben 403 en gestionar-usuarios, registrar usuarios, los cuatro endpoints, crear enlace, registrar/editar escuela y descarga de CSV de `dato-puro`.
- **Validaciones:** correo con formato inválido, dominio no permitido, correo duplicado en la BD (sin distinguir mayúsculas), duplicado dentro del lote, teléfono inválido, campos vacíos, más de 200 filas, arreglo vacío.
- **Todo o nada (C):** un lote con una fila inválida entre 50 no crea ningún usuario; un `IntegrityError` simulado revierte todo.
- **Sin duplicados ni huérfanos:** tras un lote de 50, hay 50 usuarios con rol 6 y la escuela correcta, 50 correos y 50 usuarios únicos; ninguno con `escuela` nula.
- **Logins únicos:** 50 alumnos con el mismo primer nombre generan 50 usuarios distintos.
- **Rendimiento:** 50 filas se crean en menos de 5 s y con un número acotado de consultas (`assertNumQueries` con tope).
- **Hash:** el login con la contraseña temporal funciona y deja el hash con las iteraciones normales.
- **Lista (D):** con 120 usuarios (docentes y alumnos) de la escuela del Enlace y 30 de otra escuela: solo aparecen los 120, 5 páginas de 25, la página 5 trae 20, búsqueda por nombre, por apellido, por correo y por nombre completo, filtro por rol y por estado, y los filtros persisten en los enlaces de paginación. Página fuera de rango no rompe.
- **Acciones:** el Enlace no puede desactivarse a sí mismo ni a usuarios de otra escuela; restablecer contraseña solo funciona con Docente/Alumno propios.
- **Alumno individual:** sigue funcionando con las reglas nuevas.

## 9. Verificación manual (local)
1. `python manage.py makemigrations core --check --dry-run` (no debe haber cambios) y `python manage.py test core`.
2. Con `runserver`, entrar como Enlace: registrar un docente, comprobar que sale en la lista, iniciar sesión con él (cambio forzado de contraseña) y confirmar que solo ve consulta.
3. Cargar la plantilla CSV con 50 alumnos (probar con `,` y con `;`, con y sin acentos) y medir el tiempo de la petición en las herramientas del navegador (< 5 s).
4. Meter filas con errores (correo malo, repetido, ya existente) y confirmar que se marcan sin recargar y que no se crea ninguno.
5. Con 100+ usuarios probar búsqueda, filtros y paginación; confirmar que otra escuela no se mezcla.

## 10. Estrategia de entrega
- **Fase local primero** (rama `Diego`): se desarrolla y prueba en local; no se hace push a `main` (dispara el despliegue) ni se toca producción hasta que apruebes las pruebas.
- **La implementación inicia cuando lo indiques.** Este documento solo define el plan.
- **Producción:** no hay pasos manuales nuevos (sin migraciones, sin cambios en `settings.py`, sin volúmenes). El script nuevo viaja por el `collectstatic` del despliegue.

## 11. Orden de implementación sugerido
1. `core/usuarios_escuela.py` (validación, hash ligero, creación atómica) y sus pruebas unitarias.
2. Endpoints JSON (docente, alumno individual, validar, lote) + pruebas de permisos y transacción.
3. Lista con búsqueda, filtros y paginación + acciones (activar/desactivar/restablecer).
4. Pantalla de registro con pestañas y `carga-alumnos.js` (filas dinámicas, CSV, validación en línea, modal de credenciales).
5. Ajuste de las pruebas existentes, corrida completa y verificación manual (sección 9).

## 12. Supuestos que dejo anotados (corrígelos si no aplican)
- Solo el **Enlace** registra docentes y alumnos; el administrador/superusuario no lo hace por estas rutas.
- El lote es **todo o nada**: una fila inválida impide guardar el resto (así se evita dejar registros a medias). El usuario corrige y reenvía.
- Máximo **200 filas** por lote y CSV de hasta 1 MB; solo CSV (no `.xlsx`).
- Columnas del lote: `nombre`, `apellidos`, `correo`, `telefono` (opcional). Edad, enfermedad y atleta no se capturan en el alta.
- **25 usuarios por página.**
- El “restablecer contraseña temporal” de la sección 4.5 entra como alcance mínimo de recuperación.
- Un correo ya registrado en cualquier escuela cuenta como duplicado (la unicidad es global).

## 13. Observaciones secundarias (sin acción ahora)
- El registro del Enlace (superusuario) sigue usando el hash normal de 0.68 s; es un solo usuario, no hace falta cambiarlo.
- `view_escuelas` ejecuta una consulta a Mongo por escuela en la comparativa; con muchas escuelas se notará, pero no forma parte de este plan.
- Los ids de rol (4, 5, 6) siguen usándose como números en el resto del código; el módulo nuevo los concentra en constantes sin refactorizar lo demás.
