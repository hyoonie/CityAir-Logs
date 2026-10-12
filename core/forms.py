from django import forms
from django.db.models import Q
from django.utils import timezone

from .models import DocumentoEscuela, Escuela, Estado, Municipio, Usuario
from .validators import (
    dividir_apellidos,
    normalizar_telefono,
    validar_cct_mexico,
    validar_correo_dominio,
    validar_fecha_no_futura,
    validar_pdf,
)

class UploadFileForm(forms.Form):
    file = forms.FileField(label="Cargar archivo CSV de datos")


# ──────────────────────────────────────────────
# MÓDULO ESCUELAS
# ──────────────────────────────────────────────

class BootstrapMixin:
    """Asigna las clases de Bootstrap a los widgets del formulario."""

    def aplicar_estilos(self):
        for campo in self.fields.values():
            widget = campo.widget
            if isinstance(widget, forms.RadioSelect):
                continue
            if isinstance(widget, forms.CheckboxInput):
                clase = 'form-check-input'
            elif isinstance(widget, forms.Select):
                clase = 'form-select'
            else:
                clase = 'form-control'
            widget.attrs['class'] = f"{widget.attrs.get('class', '')} {clase}".strip()


class MunicipioChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return obj.nombre


class EscuelaForm(BootstrapMixin, forms.ModelForm):
    # Campos obligatorios según el mockup (en BD son opcionales por las escuelas ya registradas)
    CAMPOS_OBLIGATORIOS = (
        'nombre', 'nivel', 'sostenimiento', 'pais', 'latitud', 'longitud',
    )
    # Estado y municipio: catálogo cerrado en México; texto libre en otros países (se validan en clean)
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
            'estado': 'Estado',
            'municipio': 'Municipio',
            'colonia': 'Colonia',
            'calle': 'Calle',
            'numero': 'Número',
            'codigo_postal': 'Código postal',
            'latitud': 'Latitud',
            'longitud': 'Longitud',
        }
        widgets = {
            'telefono': forms.TextInput(attrs={'placeholder': '228-165-0525', 'inputmode': 'tel'}),
            'correo_institucional': forms.EmailInput(attrs={'placeholder': 'contacto@escuela.edu.mx'}),
            'clave_centro_trabajo': forms.TextInput(attrs={'placeholder': '30EPR0001X', 'maxlength': 10}),
            'latitud': forms.NumberInput(attrs={'step': '0.000001', 'min': '-90', 'max': '90'}),
            'longitud': forms.NumberInput(attrs={'step': '0.000001', 'min': '-180', 'max': '180'}),
            'codigo_postal': forms.TextInput(attrs={'inputmode': 'numeric'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for nombre in self.CAMPOS_OBLIGATORIOS:
            self.fields[nombre].required = True
        # La obligatoriedad de estado/municipio depende del país: se valida en clean()
        for nombre in ('estado', 'municipio'):
            self.fields[nombre].required = False
        if not self.instance.pk:
            self.fields['pais'].initial = 'mexico'
        self._configurar_catalogo()
        self.aplicar_estilos()

    def _configurar_catalogo(self):
        """Estados activos y, solo del estado elegido, sus municipios (la lista se recarga por AJAX)."""
        actual_estado = self.instance.estado_catalogo_id
        actual_municipio = self.instance.municipio_catalogo_id
        self.fields['estado_catalogo'].queryset = Estado.objects.filter(Q(activo=True) | Q(pk=actual_estado))

        estado_id = actual_estado
        if self.is_bound:
            try:
                estado_id = int(self.data.get(self.add_prefix('estado_catalogo')) or 0) or None
            except (TypeError, ValueError):
                estado_id = None
        self.fields['municipio_catalogo'].queryset = (
            Municipio.objects.filter(estado_id=estado_id).filter(Q(activo=True) | Q(pk=actual_municipio))
            if estado_id else Municipio.objects.none()
        )

    def clean_clave_centro_trabajo(self):
        # Vacía se guarda como NULL para no chocar con la restricción única (país, CCT)
        clave = (self.cleaned_data.get('clave_centro_trabajo') or '').strip().upper()
        return clave or None

    def clean_telefono(self):
        telefono = self.cleaned_data.get('telefono')
        return normalizar_telefono(telefono) if telefono else None

    def clean_correo_institucional(self):
        correo = (self.cleaned_data.get('correo_institucional') or '').strip().lower()
        if correo:
            validar_correo_dominio(correo)
        return correo or None

    def clean_codigo_postal(self):
        cp = (self.cleaned_data.get('codigo_postal') or '').strip()
        if cp and not cp.isdigit():
            raise forms.ValidationError('El código postal solo debe contener dígitos.')
        return cp or None

    def clean_latitud(self):
        lat = self.cleaned_data.get('latitud')
        if lat is not None and not -90 <= lat <= 90:
            raise forms.ValidationError('La latitud debe estar entre -90 y 90.')
        return lat

    def clean_longitud(self):
        lng = self.cleaned_data.get('longitud')
        if lng is not None and not -180 <= lng <= 180:
            raise forms.ValidationError('La longitud debe estar entre -180 y 180.')
        return lng

    def clean(self):
        cleaned = super().clean()
        pais = cleaned.get('pais')
        clave = cleaned.get('clave_centro_trabajo')
        if clave and pais == 'mexico':
            try:
                validar_cct_mexico(clave)
            except forms.ValidationError as e:
                self.add_error('clave_centro_trabajo', e)

        if pais == 'mexico':
            self._clean_ubicacion_catalogo(cleaned)
        elif pais:
            # Fuera de México no hay catálogo: se capturan como texto
            for campo in ('estado', 'municipio'):
                if not (cleaned.get(campo) or '').strip():
                    self.add_error(campo, self.fields[campo].error_messages['required'])
            cleaned['estado_catalogo'] = None
            cleaned['municipio_catalogo'] = None
        return cleaned

    def _clean_ubicacion_catalogo(self, cleaned):
        estado = cleaned.get('estado_catalogo')
        municipio = cleaned.get('municipio_catalogo')
        requerido = self.fields['estado_catalogo'].error_messages['required']
        if not estado and 'estado_catalogo' not in self.errors:
            self.add_error('estado_catalogo', requerido)
        if not municipio and 'municipio_catalogo' not in self.errors:
            self.add_error('municipio_catalogo', requerido)
        if estado and municipio:
            if municipio.estado_id != estado.id:
                self.add_error('municipio_catalogo', 'El municipio no pertenece al estado seleccionado.')
            else:
                # Copia del nombre en los campos de texto que usan el listado, el detalle y los reportes
                cleaned['estado'] = estado.nombre
                cleaned['municipio'] = municipio.nombre


class ConvenioForm(BootstrapMixin, forms.ModelForm):
    class Meta:
        model = DocumentoEscuela
        fields = ['tipo', 'archivo', 'fecha_firma']
        labels = {
            'tipo': 'Tipo de documento',
            'archivo': 'Archivo del documento',
            'fecha_firma': 'Fecha de firma',
        }
        widgets = {
            'archivo': forms.FileInput(attrs={'accept': 'application/pdf,.pdf'}),
            'fecha_firma': forms.DateInput(attrs={'type': 'date'}, format='%Y-%m-%d'),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Se toma antes de validar: al validar, el PDF nuevo ya queda asignado a la instancia
        self.archivo_original = self.instance.archivo.name if self.instance.pk else None
        # En edición el PDF ya existe: solo se sube uno nuevo para reemplazarlo
        self.fields['archivo'].required = not (self.instance.pk and self.instance.archivo)
        self.fields['fecha_firma'].widget.attrs['max'] = timezone.localdate().isoformat()
        self.aplicar_estilos()

    def clean_archivo(self):
        archivo = self.cleaned_data.get('archivo')
        # Solo se valida el archivo recién subido, no el que ya estaba guardado
        if archivo and hasattr(archivo, 'content_type'):
            validar_pdf(archivo)
        return archivo

    def clean_fecha_firma(self):
        fecha = self.cleaned_data.get('fecha_firma')
        validar_fecha_no_futura(fecha)
        return fecha


def escuelas_sin_enlace(incluir=None):
    """Escuelas activas sin Enlace; `incluir` agrega la escuela actual del enlace en edición."""
    condicion = Q(activa=True) & ~Q(usuarios__tipousuario__id_tipo=4)
    if incluir is not None:
        condicion |= Q(pk=incluir.pk)
    return Escuela.objects.filter(condicion).order_by('nombre')


class EnlaceForm(BootstrapMixin, forms.Form):
    ACCESO_CHOICES = [
        ('invitacion', 'Enviar invitación por correo'),
        ('temporal', 'Generar contraseña temporal'),
    ]

    escuela = forms.ModelChoiceField(
        queryset=Escuela.objects.none(),
        label='Escuelas sin enlace',
        empty_label='Selecciona una escuela',
    )
    nombre = forms.CharField(label='Nombre(s)', max_length=45)
    apellidos = forms.CharField(label='Apellidos', max_length=91)
    correo = forms.EmailField(label='Correo (será su usuario)', max_length=255)
    telefono = forms.CharField(label='Teléfono', max_length=14,
                               widget=forms.TextInput(attrs={'placeholder': '228-123-4567', 'inputmode': 'tel'}))
    cargo = forms.ChoiceField(label='Cargo', choices=[('', 'Selecciona un cargo')] + Usuario.CARGO_CHOICES)
    acceso = forms.ChoiceField(choices=ACCESO_CHOICES, initial='temporal', widget=forms.RadioSelect)

    def __init__(self, *args, enlace=None, **kwargs):
        self.enlace = enlace
        super().__init__(*args, **kwargs)
        escuela_actual = enlace.escuela if enlace else None
        self.fields['escuela'].queryset = escuelas_sin_enlace(incluir=escuela_actual)
        if enlace:
            # En edición no se elige método de acceso: solo se puede restablecer la contraseña temporal
            del self.fields['acceso']
            self.fields['restablecer_password'] = forms.BooleanField(
                required=False, label='Restablecer contraseña temporal',
            )
        self.aplicar_estilos()

    def clean_nombre(self):
        return ' '.join(self.cleaned_data['nombre'].split())

    def clean_apellidos(self):
        apellidos = ' '.join(self.cleaned_data['apellidos'].split())
        paterno, materno = dividir_apellidos(apellidos)
        if len(paterno) > 45 or len(materno) > 45:
            raise forms.ValidationError('Cada apellido debe tener como máximo 45 caracteres.')
        return apellidos

    def clean_correo(self):
        correo = self.cleaned_data['correo'].strip().lower()
        validar_correo_dominio(correo)
        otros = Usuario.objects.filter(email__iexact=correo)
        if self.enlace:
            otros = otros.exclude(pk=self.enlace.pk)
        if otros.exists():
            raise forms.ValidationError('Ya existe un usuario registrado con este correo.')
        return correo

    def clean_telefono(self):
        return normalizar_telefono(self.cleaned_data['telefono'])

    def clean_acceso(self):
        acceso = self.cleaned_data['acceso']
        if acceso != 'temporal':
            raise forms.ValidationError('La invitación por correo aún no está disponible. Usa la contraseña temporal.')
        return acceso
