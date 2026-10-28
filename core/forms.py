from django import forms
from django.db import IntegrityError, transaction
from django.db.models import Q

from .models import Escuela, Estado, Municipio
from .validators import MENSAJE_CCT, MENSAJE_TELEFONO, normalizar_telefono, validar_cct_mexico

class UploadFileForm(forms.Form):
    file = forms.FileField(label="Cargar archivo CSV de datos")


# ──────────────────────────────────────────────
# MÓDULO ESCUELAS
# ──────────────────────────────────────────────

class BootstrapMixin:
    """Asigna las clases de Bootstrap a los widgets del formulario."""

    def aplicar_estilos(self):
        for campo in self.fields.values():
            clase = 'form-select' if isinstance(campo.widget, forms.Select) else 'form-control'
            campo.widget.attrs['class'] = f"{campo.widget.attrs.get('class', '')} {clase}".strip()

    def full_clean(self):
        super().full_clean()
        for nombre in self.errors:
            if nombre in self.fields:
                widget = self.fields[nombre].widget
                widget.attrs['class'] = f"{widget.attrs.get('class', '')} is-invalid".strip()


class MunicipioChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return obj.nombre


class EscuelaForm(BootstrapMixin, forms.ModelForm):
    # En BD son opcionales por las escuelas registradas antes de este formulario.
    # La CCT solo es obligatoria en México (se valida en clean()).
    CAMPOS_OBLIGATORIOS = ('nombre', 'nivel', 'sostenimiento', 'turno', 'pais', 'latitud', 'longitud')
    # Texto de la opción vacía de cada combo (en lugar del '---------' de Django)
    OPCION_VACIA = {
        'nivel': 'Selecciona el nivel educativo',
        'sostenimiento': 'Selecciona el sostenimiento',
        'turno': 'Selecciona el turno',
        'pais': 'Selecciona un país',
    }
    # Mensajes de error bajo cada campo. Los usa el servidor y se pasan al JS de la plantilla
    # para validar en el navegador con los mismos textos.
    MENSAJES = {
        'nombre': 'Ingresa el nombre oficial de la escuela.',
        'clave_centro_trabajo': 'Ingresa la clave CCT; es obligatoria para escuelas de México.',
        'nivel': 'Selecciona el nivel educativo.',
        'sostenimiento': 'Selecciona el sostenimiento.',
        'turno': 'Selecciona el turno.',
        'pais': 'Selecciona el país.',
        'estado_catalogo': 'Selecciona el estado.',
        'municipio_catalogo': 'Selecciona el municipio.',
        'estado': 'Ingresa el estado o departamento.',
        'municipio': 'Ingresa el municipio.',
        'latitud': 'Indica la latitud; puedes elegirla en el mapa.',
        'longitud': 'Indica la longitud; puedes elegirla en el mapa.',
        'cct_formato': MENSAJE_CCT,
        'cct_duplicada': 'Ya existe una escuela con esa clave CCT en este país.',
        'telefono_formato': MENSAJE_TELEFONO,
        'correo_formato': 'Ingresa un correo válido, por ejemplo contacto@escuela.edu.mx.',
        'cp_formato': 'El código postal solo debe contener dígitos.',
        'latitud_rango': 'La latitud debe estar entre -90 y 90.',
        'longitud_rango': 'La longitud debe estar entre -180 y 180.',
        'numero_formato': 'Ingresa un número válido.',
        'municipio_de_otro_estado': 'El municipio no pertenece al estado seleccionado.',
    }

    # México: catálogo INEGI. Otros países: texto libre en estado/municipio. Se valida en clean().
    estado_catalogo = forms.ModelChoiceField(
        queryset=Estado.objects.none(), required=False, label='Estado', empty_label='Selecciona un estado',
    )
    municipio_catalogo = MunicipioChoiceField(
        queryset=Municipio.objects.none(), required=False, label='Municipio', empty_label='Selecciona un municipio',
    )

    class Meta:
        model = Escuela
        fields = [
            'nombre', 'clave_centro_trabajo', 'nivel', 'sostenimiento', 'turno',
            'telefono', 'correo_institucional',
            'pais', 'estado_catalogo', 'municipio_catalogo', 'estado', 'municipio',
            'colonia', 'calle', 'numero', 'codigo_postal', 'latitud', 'longitud',
        ]
        labels = {
            'nombre': 'Nombre oficial',
            'clave_centro_trabajo': 'Clave oficial del plantel (CCT)',
            'nivel': 'Nivel educativo',
            'sostenimiento': 'Sostenimiento',
            'turno': 'Turno',
            'telefono': 'Teléfono',
            'correo_institucional': 'Correo institucional',
            'pais': 'País',
            'estado': 'Estado / departamento',
            'municipio': 'Municipio',
            'colonia': 'Colonia',
            'calle': 'Calle',
            'numero': 'Número',
            'codigo_postal': 'Código postal',
            'latitud': 'Latitud',
            'longitud': 'Longitud',
        }
        help_texts = {
            'clave_centro_trabajo': 'Única por país. Opcional si el país no tiene equivalente.',
        }
        widgets = {
            'clave_centro_trabajo': forms.TextInput(attrs={'placeholder': '30EPR0001X'}),
            'telefono': forms.TextInput(attrs={'placeholder': '228-165-0525', 'inputmode': 'tel'}),
            'correo_institucional': forms.EmailInput(attrs={'placeholder': 'contacto@escuela.edu.mx'}),
            'codigo_postal': forms.TextInput(attrs={'inputmode': 'numeric'}),
            'latitud': forms.NumberInput(attrs={'step': '0.000001', 'min': '-90', 'max': '90'}),
            'longitud': forms.NumberInput(attrs={'step': '0.000001', 'min': '-180', 'max': '180'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for nombre in self.CAMPOS_OBLIGATORIOS:
            self.fields[nombre].required = True
        for nombre, texto in self.OPCION_VACIA.items():
            opciones = [c for c in self.fields[nombre].choices if c[0] != '']
            self.fields[nombre].choices = [('', texto)] + opciones
        for nombre in ('estado', 'municipio'):
            self.fields[nombre].required = False
        for nombre, campo in self.fields.items():
            if nombre in self.MENSAJES:
                campo.error_messages['required'] = self.MENSAJES[nombre]
        self.fields['correo_institucional'].error_messages['invalid'] = self.MENSAJES['correo_formato']
        for nombre in ('latitud', 'longitud'):
            self.fields[nombre].error_messages['invalid'] = self.MENSAJES['numero_formato']
        self._configurar_catalogo()
        self.aplicar_estilos()

    def _configurar_catalogo(self):
        """Estados activos y, del estado elegido, sus municipios (en el navegador se recargan por AJAX).
        Se incluye el valor actual de la escuela aunque INEGI lo haya dado de baja."""
        estado_actual = self.instance.estado_catalogo_id
        municipio_actual = self.instance.municipio_catalogo_id
        self.fields['estado_catalogo'].queryset = Estado.objects.filter(Q(activo=True) | Q(pk=estado_actual))

        estado_id = estado_actual
        if self.is_bound:
            try:
                estado_id = int(self.data.get(self.add_prefix('estado_catalogo')) or 0) or None
            except (TypeError, ValueError):
                estado_id = None
        self.fields['municipio_catalogo'].queryset = (
            Municipio.objects.filter(estado_id=estado_id).filter(Q(activo=True) | Q(pk=municipio_actual))
            if estado_id else Municipio.objects.none()
        )

    def clean_nombre(self):
        return ' '.join(self.cleaned_data['nombre'].split())

    def clean_clave_centro_trabajo(self):
        # Vacía se guarda como NULL para no chocar con la restricción única (país, CCT)
        clave = (self.cleaned_data.get('clave_centro_trabajo') or '').strip().upper()
        return clave or None

    def clean_telefono(self):
        telefono = self.cleaned_data.get('telefono')
        return normalizar_telefono(telefono) if telefono else None

    def clean_correo_institucional(self):
        correo = (self.cleaned_data.get('correo_institucional') or '').strip().lower()
        return correo or None

    def clean_codigo_postal(self):
        cp = (self.cleaned_data.get('codigo_postal') or '').strip()
        if cp and not cp.isdigit():
            raise forms.ValidationError(self.MENSAJES['cp_formato'])
        return cp or None

    def clean_latitud(self):
        lat = self.cleaned_data.get('latitud')
        if lat is not None and not -90 <= lat <= 90:
            raise forms.ValidationError(self.MENSAJES['latitud_rango'])
        return lat

    def clean_longitud(self):
        lng = self.cleaned_data.get('longitud')
        if lng is not None and not -180 <= lng <= 180:
            raise forms.ValidationError(self.MENSAJES['longitud_rango'])
        return lng

    def clean(self):
        datos = super().clean()
        pais = datos.get('pais')
        if pais == 'mexico':
            self._clean_ubicacion_mexico(datos)
        elif pais:
            self._clean_ubicacion_texto(datos)
        self._clean_cct(datos, pais)
        return datos

    def _clean_ubicacion_mexico(self, datos):
        estado = datos.get('estado_catalogo')
        municipio = datos.get('municipio_catalogo')
        if not estado:
            self.add_error('estado_catalogo', self.MENSAJES['estado_catalogo'])
        if not municipio:
            self.add_error('municipio_catalogo', self.MENSAJES['municipio_catalogo'])
        if estado and municipio and municipio.estado_id != estado.id:
            self.add_error('municipio_catalogo', self.MENSAJES['municipio_de_otro_estado'])
            return
        if estado and municipio:
            # Copia de texto para el listado, el detalle y las vistas que no conocen el catálogo
            datos['estado'] = estado.nombre
            datos['municipio'] = municipio.nombre

    def _clean_ubicacion_texto(self, datos):
        datos['estado_catalogo'] = None
        datos['municipio_catalogo'] = None
        for campo in ('estado', 'municipio'):
            datos[campo] = ' '.join((datos.get(campo) or '').split())
            if not datos[campo]:
                self.add_error(campo, self.MENSAJES[campo])

    def _clean_cct(self, datos, pais):
        clave = datos.get('clave_centro_trabajo')
        if pais == 'mexico' and not clave and 'clave_centro_trabajo' not in self.errors:
            self.add_error('clave_centro_trabajo', self.MENSAJES['clave_centro_trabajo'])
            return
        if not clave or not pais:
            return
        if pais == 'mexico':
            try:
                validar_cct_mexico(clave)
            except forms.ValidationError as e:
                self.add_error('clave_centro_trabajo', e)
                return
        # La restricción única (país, CCT) también existe en BD; aquí se valida para mostrar el error en el campo
        duplicada = Escuela.objects.filter(pais=pais, clave_centro_trabajo=clave).exclude(pk=self.instance.pk)
        if duplicada.exists():
            self.add_error('clave_centro_trabajo', self.MENSAJES['cct_duplicada'])

    def guardar(self):
        """save() que convierte el choque con uniq_escuela_pais_cct (otra petición guardó la misma CCT
        entre la validación y el guardado) en un error del campo. Devuelve None si no se guardó."""
        try:
            with transaction.atomic():
                return self.save()
        except IntegrityError as e:
            if 'uniq_escuela_pais_cct' not in str(e):
                raise
            self.add_error('clave_centro_trabajo', self.MENSAJES['cct_duplicada'])
            widget = self.fields['clave_centro_trabajo'].widget
            widget.attrs['class'] = f"{widget.attrs.get('class', '')} is-invalid".strip()
            return None
