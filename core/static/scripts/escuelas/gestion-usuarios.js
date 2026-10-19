/* Gestión de docentes y alumnos por el Enlace. Sin dependencias externas.
   Las contraseñas temporales solo viven en esta página: se muestran una vez y se pueden descargar. */
(function (window, document) {
  'use strict';

  var SIN_CONEXION = 'No se pudo conectar con el servidor. Revisa tu conexión e inténtalo de nuevo.';

  function csrfToken() {
    var campo = document.querySelector('input[name=csrfmiddlewaretoken]');
    return campo ? campo.value : '';
  }

  // POST JSON o FormData; devuelve {status, datos}. Si no hay red, status 0.
  function enviar(url, cuerpo) {
    var esForm = cuerpo instanceof FormData;
    var cabeceras = { 'X-CSRFToken': csrfToken() };
    if (!esForm) cabeceras['Content-Type'] = 'application/json';
    return fetch(url, {
      method: 'POST', headers: cabeceras, credentials: 'same-origin',
      body: esForm ? cuerpo : JSON.stringify(cuerpo)
    }).then(function (resp) {
      return resp.json().catch(function () {
        return { errores: [{ mensaje: 'Respuesta inesperada del servidor (' + resp.status + ').' }] };
      }).then(function (datos) { return { status: resp.status, datos: datos }; });
    }).catch(function () {
      return { status: 0, datos: { errores: [{ mensaje: SIN_CONEXION }] } };
    });
  }

  function escapar(texto) {
    var div = document.createElement('div');
    div.textContent = texto == null ? '' : String(texto);
    return div.innerHTML;
  }

  function primerError(datos, porDefecto) {
    var e = (datos && datos.errores || [])[0];
    return e ? e.mensaje : porDefecto;
  }

  function gradoGrupo(grado, grupo) {
    return grado ? (grado + '° ' + (grupo || '')).trim() : '—';
  }

  function plural(n, uno, varios) { return n + ' ' + (n === 1 ? uno : varios); }

  // CSV con BOM (Excel respeta acentos) y celdas neutralizadas para que no se lean como fórmulas
  function descargarCSV(nombreArchivo, encabezados, filas) {
    function celda(v) {
      var t = v == null ? '' : String(v);
      if (/^[=+\-@\t\r]/.test(t)) t = "'" + t;
      return '"' + t.replace(/"/g, '""') + '"';
    }
    function linea(valores) { return valores.map(celda).join(','); }
    var contenido = '﻿' + [linea(encabezados)].concat(filas.map(linea)).join('\r\n') + '\r\n';
    var url = URL.createObjectURL(new Blob([contenido], { type: 'text/csv;charset=utf-8' }));
    var a = document.createElement('a');
    a.href = url;
    a.download = nombreArchivo;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function descargarCredenciales(usuarios, esAlumno) {
    var encabezados = esAlumno
      ? ['Nombre', 'Grado', 'Grupo', 'Usuario', 'Contraseña temporal']
      : ['Nombre', 'Correo', 'Usuario', 'Contraseña temporal'];
    var filas = usuarios.map(function (u) {
      return esAlumno ? [u.nombre, u.grado, u.grupo, u.usuario, u.password]
                      : [u.nombre, u.correo, u.usuario, u.password];
    });
    descargarCSV(esAlumno ? 'credenciales-alumnos.csv' : 'credenciales-docentes.csv', encabezados, filas);
  }

  // Mientras haya credenciales sin descargar, se avisa antes de salir de la página
  function avisoAlSalir() {
    var estado = { pendiente: false };
    window.addEventListener('beforeunload', function (ev) {
      if (estado.pendiente) { ev.preventDefault(); ev.returnValue = ''; }
    });
    return estado;
  }

  function conCarga(boton, texto) {
    boton.dataset.textoOriginal = boton.dataset.textoOriginal || boton.innerHTML;
    boton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>' + texto;
    boton.disabled = true;
  }

  function sinCarga(boton) {
    if (boton.dataset.textoOriginal) boton.innerHTML = boton.dataset.textoOriginal;
    boton.disabled = false;
  }

  // ── Agregar alumno / docente ──────────────────────────────────────────────

  function initAgregar() {
    var form = document.getElementById('form-usuario');
    var esAlumno = form.dataset.rol === 'alumno';
    var errorGeneral = document.getElementById('error-general');
    var seccion = document.getElementById('agregados');
    var lista = document.getElementById('lista-agregados');
    var panelTerminar = document.getElementById('panel-terminar');
    var botones = form.querySelectorAll('button[type=submit]');
    var campos = esAlumno ? ['nombre', 'apellidos', 'grado', 'grupo'] : ['nombre', 'apellidos', 'correo', 'telefono'];
    var creados = [];
    var salida = avisoAlSalir();
    var enviando = false;

    function limpiarErrores() {
      form.querySelectorAll('.is-invalid').forEach(function (c) { c.classList.remove('is-invalid'); });
      form.querySelectorAll('[data-error-para]').forEach(function (d) { d.textContent = ''; });
      errorGeneral.classList.add('d-none');
    }

    function marcar(campo, mensaje) {
      var input = campo && form.elements[campo];
      var aviso = campo && form.querySelector('[data-error-para="' + campo + '"]');
      if (input && aviso) {
        input.classList.add('is-invalid');
        aviso.textContent = mensaje;
      } else {
        errorGeneral.textContent = mensaje;
        errorGeneral.classList.remove('d-none');
      }
    }

    function validarLocal(d) {
      var e = {};
      if (!d.nombre) e.nombre = 'Falta el nombre';
      else if (d.nombre.length > 45) e.nombre = 'El nombre debe tener como máximo 45 caracteres';
      if (!d.apellidos) e.apellidos = 'Faltan apellidos';
      if (esAlumno) {
        if (!d.grado) e.grado = 'Falta el grado';
        if (!d.grupo) e.grupo = 'Falta el grupo';
      } else if (!d.correo) {
        e.correo = 'Falta el correo';
      } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(d.correo)) {
        e.correo = 'Correo no válido';
      }
      return e;
    }

    function agregarFila(u) {
      creados.push(u);
      var li = document.createElement('li');
      li.className = 'list-group-item d-flex align-items-center gap-3 py-3';
      li.innerHTML =
        '<i class="bi bi-check-square text-success"></i>' +
        '<div class="flex-grow-1">' +
          '<div>' + escapar(u.nombre) + '</div>' +
          '<div class="small text-muted">Usuario: <strong class="font-monospace">' + escapar(u.usuario) + '</strong>' +
          ' · Contraseña temporal: <strong class="font-monospace">' + escapar(u.password) + '</strong></div>' +
        '</div>' +
        '<div class="text-muted text-end small">' + escapar(esAlumno ? gradoGrupo(u.grado, u.grupo) : u.correo) + '</div>';
      lista.appendChild(li);
      seccion.classList.remove('d-none');
      salida.pendiente = true;
    }

    function bloquearFormulario(bloquear) {
      form.querySelectorAll('input:not([type=hidden]), select, button').forEach(function (c) { c.disabled = bloquear; });
      panelTerminar.classList.toggle('d-none', !bloquear);
    }

    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      if (enviando) return;
      var terminar = ev.submitter && ev.submitter.dataset.accion === 'terminar';
      limpiarErrores();

      var datos = {};
      campos.forEach(function (c) { datos[c] = form.elements[c].value.trim(); });
      var locales = validarLocal(datos);
      if (Object.keys(locales).length) {
        Object.keys(locales).forEach(function (c) { marcar(c, locales[c]); });
        form.querySelector('.is-invalid').focus();
        return;
      }

      enviando = true;
      botones.forEach(function (b) { b.disabled = true; });
      enviar(form.dataset.url, datos).then(function (r) {
        enviando = false;
        botones.forEach(function (b) { b.disabled = false; });
        if (r.status === 201) {
          r.datos.usuarios.forEach(agregarFila);
          // Grado y grupo se conservan para capturar seguido al mismo grupo
          ['nombre', 'apellidos', 'correo', 'telefono'].forEach(function (c) {
            if (form.elements[c]) form.elements[c].value = '';
          });
          if (terminar) {
            bloquearFormulario(true);
            seccion.scrollIntoView({ behavior: 'smooth' });
          } else {
            form.elements.nombre.focus();
          }
          return;
        }
        var errores = r.datos.errores || [];
        if (!errores.length) marcar(null, 'No se pudo guardar. Inténtalo de nuevo.');
        errores.forEach(function (e) { marcar(e.campo, e.mensaje); });
        var invalido = form.querySelector('.is-invalid');
        if (invalido) invalido.focus();
      });
    });

    document.getElementById('btn-agregar-otro').addEventListener('click', function () {
      bloquearFormulario(false);
      form.elements.nombre.focus();
    });

    document.getElementById('btn-descargar-credenciales').addEventListener('click', function () {
      descargarCredenciales(creados, esAlumno);
      salida.pendiente = false;
    });
  }

  // ── Cargar grupo desde Excel ──────────────────────────────────────────────

  function initCargaExcel() {
    var raiz = document.getElementById('carga-excel');
    var maxBytes = parseInt(raiz.dataset.maxBytes, 10);
    var input = document.getElementById('archivo');
    var zona = document.getElementById('zona-archivo');
    var chip = document.getElementById('chip-archivo');
    var errorArchivo = document.getElementById('error-archivo');
    var paso3 = document.getElementById('paso-revisar');
    var revision = document.getElementById('bloque-revision');
    var resultado = document.getElementById('bloque-resultado');
    var tbody = document.getElementById('tabla-previa');
    var btnImportar = document.getElementById('btn-importar');
    var btnErrores = document.getElementById('btn-descargar-errores');
    var errorImportar = document.getElementById('error-importar');
    var salida = avisoAlSalir();
    var vista = null;
    var nombreArchivo = '';
    var creados = [];

    function mostrarErrorArchivo(mensaje) {
      errorArchivo.textContent = mensaje;
      errorArchivo.classList.remove('d-none');
    }

    function reiniciar() {
      vista = null;
      creados = [];
      input.value = '';
      chip.classList.add('d-none');
      zona.classList.remove('d-none');
      errorArchivo.classList.add('d-none');
      paso3.classList.add('d-none');
      salida.pendiente = false;
    }

    function procesar(archivo) {
      errorArchivo.classList.add('d-none');
      if (!/\.xlsx$/i.test(archivo.name)) return mostrarErrorArchivo('El archivo debe tener formato .xlsx.');
      if (archivo.size > maxBytes) {
        return mostrarErrorArchivo('El archivo no debe superar ' + Math.round(maxBytes / 1048576) + ' MB.');
      }
      var datos = new FormData();
      datos.append('archivo', archivo);
      zona.classList.add('gu-cargando');
      enviar(raiz.dataset.urlPrevia, datos).then(function (r) {
        zona.classList.remove('gu-cargando');
        input.value = '';
        if (r.status !== 200) return mostrarErrorArchivo(primerError(r.datos, 'No se pudo leer el archivo.'));
        vista = r.datos;
        nombreArchivo = archivo.name;
        document.getElementById('chip-nombre').textContent = archivo.name;
        document.getElementById('chip-filas').textContent = '· ' + plural(vista.filas.length, 'fila', 'filas');
        zona.classList.add('d-none');
        chip.classList.remove('d-none');
        mostrarRevision();
      });
    }

    function mostrarRevision() {
      document.getElementById('cont-listas').textContent = plural(vista.listas, 'lista', 'listas');
      document.getElementById('cont-errores').textContent = vista.con_errores + ' con errores';
      document.getElementById('cont-errores-wrap').classList.toggle('d-none', vista.con_errores === 0);
      tbody.innerHTML = vista.filas.map(function (f) {
        var conError = f.errores.length > 0;
        return '<tr class="' + (conError ? 'gu-fila-error' : '') + '">' +
          '<td>' + f.fila + '</td>' +
          '<td>' + escapar(f.nombre) + '</td>' +
          '<td>' + escapar(f.apellidos) + '</td>' +
          '<td>' + (conError ? escapar(f.grado) : escapar(f.grado) + '°') + '</td>' +
          '<td>' + escapar(f.grupo) + '</td>' +
          '<td class="' + (conError ? 'text-danger' : 'text-success') + '">' +
            (conError ? escapar(f.errores.join('. ')) : 'Lista') + '</td>' +
          '</tr>';
      }).join('');
      btnImportar.dataset.textoOriginal = '';
      btnImportar.textContent = 'Importar ' + plural(vista.listas, 'alumno', 'alumnos');
      btnImportar.disabled = vista.listas === 0;
      btnErrores.classList.toggle('d-none', vista.con_errores === 0);
      errorImportar.classList.add('d-none');
      revision.classList.remove('d-none');
      resultado.classList.add('d-none');
      paso3.classList.remove('d-none');
    }

    function mostrarResultado(usuarios) {
      creados = usuarios;
      salida.pendiente = true;
      document.getElementById('titulo-resultado').textContent =
        'Se ' + (usuarios.length === 1 ? 'registró 1 alumno' : 'registraron ' + usuarios.length + ' alumnos');
      document.getElementById('tabla-credenciales').innerHTML = usuarios.map(function (u) {
        return '<tr><td>' + escapar(u.nombre) + '</td><td>' + escapar(gradoGrupo(u.grado, u.grupo)) + '</td>' +
          '<td class="font-monospace">' + escapar(u.usuario) + '</td>' +
          '<td class="font-monospace">' + escapar(u.password) + '</td></tr>';
      }).join('');
      document.getElementById('btn-quitar-archivo').disabled = true;
      revision.classList.add('d-none');
      resultado.classList.remove('d-none');
      resultado.scrollIntoView({ behavior: 'smooth' });
    }

    // Selección y arrastre del archivo
    document.getElementById('btn-seleccionar').addEventListener('click', function () { input.click(); });
    input.addEventListener('change', function () { if (input.files[0]) procesar(input.files[0]); });
    ['dragenter', 'dragover'].forEach(function (evento) {
      zona.addEventListener(evento, function (ev) { ev.preventDefault(); zona.classList.add('gu-arrastrando'); });
    });
    ['dragleave', 'drop'].forEach(function (evento) {
      zona.addEventListener(evento, function (ev) { ev.preventDefault(); zona.classList.remove('gu-arrastrando'); });
    });
    zona.addEventListener('drop', function (ev) {
      var archivo = ev.dataTransfer.files[0];
      if (archivo) procesar(archivo);
    });

    document.getElementById('btn-quitar-archivo').addEventListener('click', reiniciar);
    document.getElementById('btn-cancelar').addEventListener('click', reiniciar);
    document.getElementById('btn-otro-archivo').addEventListener('click', function () {
      if (salida.pendiente && !window.confirm('No has descargado las credenciales. ¿Cargar otro archivo de todos modos?')) return;
      document.getElementById('btn-quitar-archivo').disabled = false;
      reiniciar();
    });

    btnErrores.addEventListener('click', function () {
      var filas = vista.filas.filter(function (f) { return f.errores.length; }).map(function (f) {
        return [f.fila, f.nombre, f.apellidos, f.grado, f.grupo, f.errores.join('. ')];
      });
      descargarCSV('errores-' + nombreArchivo.replace(/\.xlsx$/i, '') + '.csv',
                   ['Fila', 'Nombre(s)', 'Apellidos', 'Grado', 'Grupo', 'Error'], filas);
    });

    btnImportar.addEventListener('click', function () {
      if (!vista || btnImportar.disabled) return;
      var filas = vista.filas.filter(function (f) { return !f.errores.length; }).map(function (f) {
        return { fila: f.fila, nombre: f.nombre, apellidos: f.apellidos, grado: f.grado, grupo: f.grupo };
      });
      errorImportar.classList.add('d-none');
      conCarga(btnImportar, 'Importando…');
      enviar(raiz.dataset.urlImportar, { filas: filas }).then(function (r) {
        if (r.status === 201) return mostrarResultado(r.datos.usuarios);
        sinCarga(btnImportar);
        var errores = r.datos.errores || [];
        var detalle = errores.filter(function (e) { return e.fila; }).map(function (e) {
          return 'Fila ' + e.fila + ': ' + e.mensaje;
        });
        errorImportar.innerHTML = escapar(detalle.length
          ? 'No se registró a nadie. Algunas filas ya no son válidas; vuelve a cargar el archivo.'
          : primerError(r.datos, 'No se pudo importar. Inténtalo de nuevo.')) +
          (detalle.length ? '<ul class="mb-0 mt-1">' + detalle.map(function (d) { return '<li>' + escapar(d) + '</li>'; }).join('') + '</ul>' : '');
        errorImportar.classList.remove('d-none');
      });
    });

    document.getElementById('btn-descargar-credenciales').addEventListener('click', function () {
      descargarCredenciales(creados, true);
      salida.pendiente = false;
    });
  }

  // ── Mi escuela (lista) ────────────────────────────────────────────────────

  function initLista() {
    var filtros = document.getElementById('form-filtros');
    filtros.querySelectorAll('select').forEach(function (s) {
      s.addEventListener('change', function () { filtros.submit(); });
    });

    document.querySelectorAll('form[data-confirmar]').forEach(function (f) {
      f.addEventListener('submit', function (ev) {
        if (!window.confirm(f.dataset.confirmar)) ev.preventDefault();
      });
    });

    var panel = document.getElementById('panel-credenciales');
    document.querySelectorAll('.btn-restablecer').forEach(function (boton) {
      boton.addEventListener('click', function () {
        if (!window.confirm('¿Generar una nueva contraseña temporal para ' + boton.dataset.nombre +
                            '? La anterior dejará de funcionar.')) return;
        boton.disabled = true;
        enviar(boton.dataset.url, {}).then(function (r) {
          boton.disabled = false;
          if (r.status !== 200) return window.alert(primerError(r.datos, 'No se pudo restablecer la contraseña.'));
          var u = r.datos.usuario;
          document.getElementById('cred-nombre').textContent = u.nombre;
          document.getElementById('cred-usuario').textContent = u.usuario;
          document.getElementById('cred-password').textContent = u.password;
          panel.classList.remove('d-none');
          panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
      });
    });
    document.getElementById('btn-cerrar-credenciales').addEventListener('click', function () {
      document.getElementById('cred-password').textContent = '';
      panel.classList.add('d-none');
    });
  }

  window.GestionUsuarios = { initAgregar: initAgregar, initCargaExcel: initCargaExcel, initLista: initLista };
})(window, document);
