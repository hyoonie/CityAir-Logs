/**
 * carga-alumnos.js
 * Lógica del frontend para la pantalla de registro de usuarios (rol Enlace).
 *
 * Responsabilidades:
 *  1. Formularios individuales: Docente y Alumno (validación inline + fetch JSON).
 *  2. Tabla editable de carga masiva: filas dinámicas, validación inline, envío.
 *  3. Carga de CSV con FileReader (UTF-8 ± BOM, sep , o ;, headers flexibles).
 *  4. Descarga de credenciales como Blob CSV en el navegador.
 *
 * Ninguna lógica de dominio está duplicada a mano: los dominios de correo
 * aceptados se leen del elemento <script id="dominios-data"> inyectado por Django.
 */
(function () {
    'use strict';

    // ── Utilidades generales ─────────────────────────────────────────────────

    function getCsrf() {
        const m = document.cookie.match(/csrftoken=([^;]+)/);
        return m ? m[1] : '';
    }

    function sinAcentos(s) {
        return (s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    }

    /** Lee los dominios permitidos inyectados por Django con json_script. */
    function cargarDominios() {
        try {
            const el = document.getElementById('dominios-data');
            return el ? JSON.parse(el.textContent) : { permitidos: [], con_subdominios: [] };
        } catch (e) {
            return { permitidos: [], con_subdominios: [] };
        }
    }

    const DOMINIOS = cargarDominios();
    const MAX_FILAS = 200;
    const MAX_CSV_BYTES = 1 * 1024 * 1024;  // 1 MB

    /** Valida el dominio de un correo contra la lista blanca del servidor. */
    function dominioPermitido(correo) {
        const partes = (correo || '').split('@');
        if (partes.length !== 2) return false;
        const dominio = partes[1].toLowerCase();
        if (DOMINIOS.permitidos.includes(dominio)) return true;
        return DOMINIOS.con_subdominios.some(function (d) {
            return dominio === d || dominio.endsWith('.' + d);
        });
    }

    // ── Validación de campos individuales ────────────────────────────────────

    function validarCampo(nombre, valor) {
        valor = (valor || '').trim();
        switch (nombre) {
            case 'nombre':
                if (!valor) return 'El nombre es obligatorio.';
                if (valor.length > 45) return 'Máximo 45 caracteres.';
                return null;
            case 'apellidos':
                if (!valor) return 'Los apellidos son obligatorios.';
                if (valor.length > 91) return 'Máximo 91 caracteres.';
                return null;
            case 'correo':
                if (!valor) return 'El correo es obligatorio.';
                if (valor.length > 255) return 'Máximo 255 caracteres.';
                if (!valor.includes('@')) return 'Formato de correo inválido.';
                if (!dominioPermitido(valor)) return 'Dominio no permitido.';
                return null;
            case 'telefono':
                if (!valor) return null;  // opcional
                if (!/^\d{3}[\s-]?\d{3}[\s-]?\d{4}$/.test(valor)) {
                    return 'Formato: ###-###-####.';
                }
                return null;
            default:
                return null;
        }
    }

    // ── Formularios individuales (Docente y Alumno) ──────────────────────────

    function iniciarFormularioIndividual(cfg) {
        const form    = document.getElementById(cfg.formId);
        const btn     = document.getElementById(cfg.btnId);
        const spin    = document.getElementById(cfg.spinId);
        const alerta  = document.getElementById(cfg.alertaId);
        if (!form) return;

        const campos = ['nombre', 'apellidos', 'correo', 'telefono'];

        // Validación inline al salir de cada campo
        campos.forEach(function (nombre) {
            const input = form.querySelector('[name="' + nombre + '"]');
            if (!input) return;
            const errDiv = document.getElementById(cfg.idPrefix + '-' + nombre + '-error');
            input.addEventListener('blur', function () {
                const msg = validarCampo(nombre, input.value);
                mostrarErrorCampo(input, errDiv, msg);
            });
        });

        // Correos duplicados en tabla lote (si ya hay filas activas)
        form.querySelector('[name="correo"]') && form.querySelector('[name="correo"]').addEventListener('blur', function () {
            /* Sin acción adicional para el form individual; la BD lo revisa al enviar */
        });

        form.addEventListener('submit', function (e) {
            e.preventDefault();
            alerta.innerHTML = '';

            // Validar todos los campos antes de enviar
            let hayError = false;
            campos.forEach(function (nombre) {
                const input = form.querySelector('[name="' + nombre + '"]');
                if (!input) return;
                const errDiv = document.getElementById(cfg.idPrefix + '-' + nombre + '-error');
                const msg = validarCampo(nombre, input.value);
                mostrarErrorCampo(input, errDiv, msg);
                if (msg) hayError = true;
            });
            if (hayError) return;

            btn.disabled = true;
            spin.classList.remove('d-none');

            const payload = {
                nombre:    form.querySelector('[name="nombre"]').value.trim(),
                apellidos: form.querySelector('[name="apellidos"]').value.trim(),
                correo:    form.querySelector('[name="correo"]').value.trim().toLowerCase(),
                telefono:  form.querySelector('[name="telefono"]').value.trim(),
            };

            fetch(cfg.url, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCsrf(),
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin',
                body: JSON.stringify(payload),
            })
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
            .then(function (res) {
                if (!res.ok) {
                    const errores = res.data.errores || {};
                    let msgs = [];
                    Object.keys(errores).forEach(function (campo) {
                        const input = form.querySelector('[name="' + campo + '"]');
                        const errDiv = document.getElementById(cfg.idPrefix + '-' + campo + '-error');
                        const msg = Array.isArray(errores[campo]) ? errores[campo][0] : errores[campo];
                        if (input && errDiv) {
                            mostrarErrorCampo(input, errDiv, msg);
                        } else {
                            msgs.push(msg);
                        }
                    });
                    if (res.data.error) msgs.push(res.data.error);
                    if (msgs.length) {
                        alerta.innerHTML = '<div class="alert alert-danger">' + msgs.join('<br>') + '</div>';
                    }
                    return;
                }
                // Éxito: mostrar credencial en alert
                const c = res.data.creado;
                alerta.innerHTML =
                    '<div class="alert alert-success">' +
                    '<strong>Usuario registrado.</strong><br>' +
                    'Usuario: <code>' + c.usuario + '</code><br>' +
                    'Contraseña temporal: <code class="fw-bold">' + c.password + '</code><br>' +
                    '<small class="text-muted">Anota estas credenciales. No se volverán a mostrar.</small>' +
                    '</div>';
                form.reset();
                campos.forEach(function (n) {
                    const input = form.querySelector('[name="' + n + '"]');
                    const errDiv = document.getElementById(cfg.idPrefix + '-' + n + '-error');
                    if (input) input.classList.remove('is-valid', 'is-invalid');
                    if (errDiv) errDiv.classList.add('d-none');
                });
            })
            .catch(function (e) {
                alerta.innerHTML = '<div class="alert alert-danger">Error de red: ' + e.message + '</div>';
            })
            .finally(function () {
                btn.disabled = false;
                spin.classList.add('d-none');
            });
        });
    }

    function mostrarErrorCampo(input, errDiv, mensaje) {
        if (!input) return;
        if (mensaje) {
            input.classList.add('is-invalid');
            input.classList.remove('is-valid');
            input.setAttribute('aria-invalid', 'true');
            if (errDiv) { errDiv.textContent = mensaje; errDiv.classList.remove('d-none'); }
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
            input.removeAttribute('aria-invalid');
            if (errDiv) errDiv.classList.add('d-none');
        }
    }

    // ── Tabla de lote ─────────────────────────────────────────────────────────

    const tbody        = document.getElementById('tbody-lote');
    const contador     = document.getElementById('contador-filas');
    const btnAgregar   = document.getElementById('btn-agregar-fila');
    const btnEliminar  = document.getElementById('btn-eliminar-fila');
    const btnValidar   = document.getElementById('btn-validar');
    const btnEnviar    = document.getElementById('btn-enviar-lote');
    const spinLote     = document.getElementById('spin-lote');
    const alertaLote   = document.getElementById('alerta-lote');
    const inputCsv     = document.getElementById('input-csv');
    const btnCargarCsv = document.getElementById('btn-cargar-csv');

    let filaCounter = 0;

    function contarFilas() {
        return tbody ? tbody.querySelectorAll('tr').length : 0;
    }

    function actualizarContador() {
        if (contador) contador.textContent = contarFilas() + ' / ' + MAX_FILAS + ' filas';
    }

    function crearFila(datos) {
        datos = datos || {};
        filaCounter++;
        const idx = filaCounter;
        const campos = ['nombre', 'apellidos', 'correo', 'telefono'];
        const tr = document.createElement('tr');
        tr.dataset.fila = idx;

        // Número
        const tdNum = document.createElement('td');
        tdNum.className = 'text-muted text-center small';
        tdNum.textContent = contarFilas() + 1;
        tr.appendChild(tdNum);

        campos.forEach(function (c) {
            const td = document.createElement('td');
            const wrapper = document.createElement('div');
            const input = document.createElement('input');
            input.type = c === 'correo' ? 'email' : 'text';
            input.className = 'form-control form-control-sm';
            input.name = c;
            input.maxLength = c === 'correo' ? 255 : (c === 'apellidos' ? 91 : (c === 'nombre' ? 45 : 12));
            input.placeholder = c === 'correo' ? 'correo@escuela.edu.mx' : '';
            if (datos[c]) input.value = datos[c];
            if (c !== 'telefono') input.required = true;

            const errDiv = document.createElement('div');
            errDiv.className = 'field-error d-none';
            errDiv.setAttribute('role', 'alert');

            input.addEventListener('blur', function () {
                const msg = validarCampo(c, input.value);
                if (c === 'correo' && !msg) validarCorreosDuplicadosEnTabla();
                mostrarErrorCampo(input, errDiv, msg);
            });

            wrapper.appendChild(input);
            wrapper.appendChild(errDiv);
            td.appendChild(wrapper);
            tr.appendChild(td);
        });

        return tr;
    }

    function agregarFila(datos) {
        if (!tbody) return;
        if (contarFilas() >= MAX_FILAS) {
            alertaLote.innerHTML = '<div class="alert alert-warning">Máximo ' + MAX_FILAS + ' filas.</div>';
            return;
        }
        tbody.appendChild(crearFila(datos));
        actualizarContador();
        renumerarFilas();
    }

    function renumerarFilas() {
        if (!tbody) return;
        tbody.querySelectorAll('tr').forEach(function (tr, i) {
            const td = tr.querySelector('td:first-child');
            if (td) td.textContent = i + 1;
        });
    }

    function validarCorreosDuplicadosEnTabla() {
        if (!tbody) return;
        const inputs = tbody.querySelectorAll('input[name="correo"]');
        const vistos = {};
        inputs.forEach(function (inp) {
            const correo = inp.value.trim().toLowerCase();
            if (correo) {
                vistos[correo] = (vistos[correo] || 0) + 1;
            }
        });
        inputs.forEach(function (inp) {
            const correo = inp.value.trim().toLowerCase();
            const td = inp.closest('td');
            const errDiv = td ? td.querySelector('.field-error') : null;
            if (correo && vistos[correo] > 1) {
                mostrarErrorCampo(inp, errDiv, 'Correo repetido en la tabla.');
            } else if (correo && !validarCampo('correo', correo)) {
                mostrarErrorCampo(inp, errDiv, null);
            }
        });
    }

    function leerFilasTabla() {
        if (!tbody) return [];
        return Array.from(tbody.querySelectorAll('tr')).map(function (tr) {
            return {
                nombre:    tr.querySelector('input[name="nombre"]').value.trim(),
                apellidos: tr.querySelector('input[name="apellidos"]').value.trim(),
                correo:    tr.querySelector('input[name="correo"]').value.trim().toLowerCase(),
                telefono:  tr.querySelector('input[name="telefono"]').value.trim(),
            };
        });
    }

    function pintarErroresEnTabla(errores) {
        if (!tbody) return;
        // Limpiar errores previos del servidor
        tbody.querySelectorAll('input').forEach(function (inp) {
            inp.classList.remove('is-invalid');
            const errDiv = inp.closest('td') && inp.closest('td').querySelector('.field-error');
            if (errDiv) errDiv.classList.add('d-none');
        });

        errores.forEach(function (err) {
            if (!err.fila || err.fila < 1) return;
            const tr = tbody.querySelectorAll('tr')[err.fila - 1];
            if (!tr) return;
            const input = tr.querySelector('input[name="' + err.campo + '"]');
            const errDiv = input && input.closest('td') && input.closest('td').querySelector('.field-error');
            if (input && errDiv) {
                mostrarErrorCampo(input, errDiv, err.mensaje);
            }
        });
    }

    if (btnAgregar) btnAgregar.addEventListener('click', function () { agregarFila(); });

    if (btnEliminar) btnEliminar.addEventListener('click', function () {
        if (!tbody) return;
        const filas = tbody.querySelectorAll('tr');
        if (filas.length > 0) {
            tbody.removeChild(filas[filas.length - 1]);
            actualizarContador();
        }
    });

    // ── Carga de CSV ──────────────────────────────────────────────────────────

    if (btnCargarCsv) btnCargarCsv.addEventListener('click', function () {
        if (!inputCsv || !inputCsv.files || !inputCsv.files[0]) {
            alertaLote.innerHTML = '<div class="alert alert-warning">Selecciona un archivo CSV.</div>';
            return;
        }
        const archivo = inputCsv.files[0];
        if (archivo.size > MAX_CSV_BYTES) {
            alertaLote.innerHTML = '<div class="alert alert-danger">El archivo supera 1 MB.</div>';
            return;
        }
        const reader = new FileReader();
        reader.onload = function (ev) {
            try {
                let texto = ev.target.result;
                // Eliminar BOM UTF-8 si existe
                if (texto.charCodeAt(0) === 0xFEFF) texto = texto.slice(1);

                const lineas = texto.split(/\r?\n/).filter(function (l) { return l.trim() !== ''; });
                if (lineas.length < 2) {
                    alertaLote.innerHTML = '<div class="alert alert-warning">El CSV no tiene filas de datos.</div>';
                    return;
                }

                // Detectar separador: contar , vs ; en la primera línea
                const primera = lineas[0];
                const sep = (primera.split(';').length > primera.split(',').length) ? ';' : ',';

                // Mapear headers (sin acentos, minúsculas)
                const headers = primera.split(sep).map(function (h) { return sinAcentos(h.trim()); });
                const idxNombre    = headers.findIndex(function (h) { return h === 'nombre'; });
                const idxApellidos = headers.findIndex(function (h) { return h === 'apellidos'; });
                const idxCorreo    = headers.findIndex(function (h) { return h === 'correo'; });
                const idxTel       = headers.findIndex(function (h) { return h === 'telefono'; });

                if (idxNombre < 0 || idxApellidos < 0 || idxCorreo < 0) {
                    alertaLote.innerHTML =
                        '<div class="alert alert-danger">El CSV debe tener columnas: ' +
                        '<code>nombre, apellidos, correo</code> (y opcionalmente <code>telefono</code>).</div>';
                    return;
                }

                // Limpiar tabla actual
                if (tbody) tbody.innerHTML = '';
                filaCounter = 0;
                alertaLote.innerHTML = '';
                let importadas = 0;

                lineas.slice(1).forEach(function (linea) {
                    const celdas = linea.split(sep);
                    if (celdas.every(function (c) { return c.trim() === ''; })) return;
                    const datos = {
                        nombre:    (celdas[idxNombre]    || '').trim(),
                        apellidos: (celdas[idxApellidos] || '').trim(),
                        correo:    (celdas[idxCorreo]    || '').trim().toLowerCase(),
                        telefono:  idxTel >= 0 ? (celdas[idxTel] || '').trim() : '',
                    };
                    agregarFila(datos);
                    importadas++;
                });

                alertaLote.innerHTML =
                    '<div class="alert alert-success">' + importadas + ' fila(s) cargada(s) desde CSV.</div>';
                actualizarContador();
            } catch (e) {
                alertaLote.innerHTML = '<div class="alert alert-danger">Error al leer el CSV: ' + e.message + '</div>';
            }
        };
        reader.readAsText(archivo, 'utf-8');
    });

    // ── Validar (sin guardar) ─────────────────────────────────────────────────

    function urlsLote() {
        // Las URLs se leen desde los atributos data del botón de envío
        const btnE = document.getElementById('btn-enviar-lote');
        return btnE
            ? { validar: btnE.dataset.urlValidar, lote: btnE.dataset.urlLote }
            : {};
    }

    if (btnValidar) btnValidar.addEventListener('click', function () {
        alertaLote.innerHTML = '';
        const filas = leerFilasTabla();
        if (!filas.length) {
            alertaLote.innerHTML = '<div class="alert alert-warning">Agrega al menos una fila.</div>';
            return;
        }

        btnValidar.disabled = true;
        const urls = urlsLote();

        fetch(urls.validar, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCsrf(), 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ filas: filas }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.errores && data.errores.length) {
                pintarErroresEnTabla(data.errores);
                alertaLote.innerHTML =
                    '<div class="alert alert-danger"><i class="bi bi-exclamation-circle me-1"></i>' +
                    data.errores.length + ' error(es) encontrado(s). Corrígelos antes de enviar.</div>';
            } else {
                alertaLote.innerHTML =
                    '<div class="alert alert-success"><i class="bi bi-check-circle me-1"></i>' +
                    'Validación exitosa. Puedes registrar a los alumnos.</div>';
            }
        })
        .catch(function (e) {
            alertaLote.innerHTML = '<div class="alert alert-danger">Error de red: ' + e.message + '</div>';
        })
        .finally(function () { btnValidar.disabled = false; });
    });

    // ── Enviar lote ───────────────────────────────────────────────────────────

    if (btnEnviar) btnEnviar.addEventListener('click', function () {
        alertaLote.innerHTML = '';
        const filas = leerFilasTabla();
        if (!filas.length) {
            alertaLote.innerHTML = '<div class="alert alert-warning">Agrega al menos una fila.</div>';
            return;
        }
        if (!confirm('¿Registrar ' + filas.length + ' alumno(s)? Esta acción no se puede deshacer.')) return;

        btnEnviar.disabled = true;
        if (spinLote) spinLote.classList.remove('d-none');
        const urls = urlsLote();

        fetch(urls.lote, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCsrf(), 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ filas: filas }),
        })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, status: r.status, data: d }; }); })
        .then(function (res) {
            if (!res.ok) {
                if (res.data.errores) {
                    pintarErroresEnTabla(res.data.errores);
                    alertaLote.innerHTML =
                        '<div class="alert alert-danger">Se encontraron errores. ' +
                        'Corrígelos y vuelve a intentar.</div>';
                } else {
                    alertaLote.innerHTML =
                        '<div class="alert alert-danger">' + (res.data.error || 'Error al registrar.') + '</div>';
                }
                return;
            }
            // Éxito: mostrar modal de credenciales
            mostrarModalCredenciales(res.data.usuarios);
            if (tbody) tbody.innerHTML = '';
            filaCounter = 0;
            actualizarContador();
            alertaLote.innerHTML =
                '<div class="alert alert-success">' + res.data.creados + ' alumno(s) registrado(s) correctamente.</div>';
        })
        .catch(function (e) {
            alertaLote.innerHTML = '<div class="alert alert-danger">Error de red: ' + e.message + '</div>';
        })
        .finally(function () {
            btnEnviar.disabled = false;
            if (spinLote) spinLote.classList.add('d-none');
        });
    });

    // ── Modal de credenciales ─────────────────────────────────────────────────

    let _credencialesActuales = [];

    function mostrarModalCredenciales(usuarios) {
        _credencialesActuales = usuarios;
        const tbody = document.getElementById('tbody-credenciales');
        if (!tbody) return;
        tbody.innerHTML = '';
        usuarios.forEach(function (u) {
            const tr = document.createElement('tr');
            tr.innerHTML =
                '<td>' + u.fila + '</td>' +
                '<td>' + escHtml(u.nombre) + '</td>' +
                '<td>' + escHtml(u.correo) + '</td>' +
                '<td><code>' + escHtml(u.usuario) + '</code></td>' +
                '<td class="fw-bold font-monospace">' + escHtml(u.password) + '</td>';
            tbody.appendChild(tr);
        });
        const modal = new bootstrap.Modal(document.getElementById('modalCredenciales'));
        modal.show();
    }

    function escHtml(s) {
        return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // ── Descarga de credenciales como CSV (Blob) ──────────────────────────────

    function neutralizarCelda(val) {
        // Prevenir inyección de fórmulas en Excel
        const s = String(val || '');
        return /^[=+\-@]/.test(s) ? "'" + s : s;
    }

    function filaCSV(celdas) {
        return celdas.map(function (c) {
            const s = neutralizarCelda(c);
            // Comillas si contiene coma, comilla o salto de línea
            return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
        }).join(',');
    }

    const btnDescargar = document.getElementById('btn-descargar-cred');
    if (btnDescargar) btnDescargar.addEventListener('click', function () {
        if (!_credencialesActuales.length) return;
        const lineas = [filaCSV(['fila', 'nombre', 'correo', 'usuario', 'contraseña_temporal'])];
        _credencialesActuales.forEach(function (u) {
            lineas.push(filaCSV([u.fila, u.nombre, u.correo, u.usuario, u.password]));
        });
        const bom  = '\ufeff';  // BOM para Excel
        const blob = new Blob([bom + lineas.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = 'credenciales_alumnos.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // ── Inicialización ────────────────────────────────────────────────────────

    // URLs leídas desde atributos data del botón enviar
    // (El template las inyecta en data-url-validar y data-url-lote)
    function init() {
        // Formulario Docente
        iniciarFormularioIndividual({
            formId:   'form-docente',
            btnId:    'btn-docente',
            spinId:   'spin-docente',
            alertaId: 'alerta-docente',
            idPrefix: 'doc',
            url:      document.getElementById('btn-docente') &&
                      document.getElementById('btn-docente').dataset.url,
        });

        // Formulario Alumno individual
        iniciarFormularioIndividual({
            formId:   'form-alumno',
            btnId:    'btn-alumno',
            spinId:   'spin-alumno',
            alertaId: 'alerta-alumno',
            idPrefix: 'alu',
            url:      document.getElementById('btn-alumno') &&
                      document.getElementById('btn-alumno').dataset.url,
        });

        // Agregar 3 filas vacías al arrancar
        if (tbody) {
            for (let i = 0; i < 3; i++) agregarFila();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

}());
