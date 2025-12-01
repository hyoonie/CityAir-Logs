from rest_framework import serializers
from .models import *

# Tabla: tipousuario
class TipoUsuarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = TipoUsuario
        fields = '__all__'

# Tabla: Enfermedad
class EnfermedadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Enfermedad
        fields = '__all__'

# Tabla: dispositivos
class DispositivoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dispositivo
        fields = '__all__'

# Tabla: Lecturas
class LecturasSerializer(serializers.ModelSerializer):
    """
    Serializador para crear y actualizar Lecturass.
    Representa el dispositivo relacionado solo con su ID.
    """
    class Meta:
        model = Lecturas
        fields = '__all__'

# Tabla: Lecturas (detalles)
class LecturasDetailSerializer(serializers.ModelSerializer):
    dispositivo = DispositivoSerializer(read_only=True)

    class Meta:
        model = Lecturas
        fields = '__all__'

# Tabla: Usuario
class UsuarioSerializer(serializers.ModelSerializer):
    tipousuario = TipoUsuarioSerializer(read_only=True)
    enfermedad = EnfermedadSerializer(read_only=True)
    
    # Campos adicionales para permitir la escritura de las relaciones a través de su ID.
    tipousuario_id = serializers.PrimaryKeyRelatedField(
        queryset=TipoUsuario.objects.all(), source='tipousuario', write_only=True, required=False
    )
    enfermedad_id = serializers.PrimaryKeyRelatedField(
        queryset=Enfermedad.objects.all(), source='enfermedad', write_only=True, required=False
    )

    class Meta:
        model = Usuario
        # Lista explícita de campos que queremos exponer en la API.
        fields = [
            'id',
            'usuario',
            'email',
            'nombre',
            'aPaterno',
            'aMaterno',
            'edad',
            'atleta',
            'tipousuario',
            'enfermedad',
            'tipousuario_id',
            'enfermedad_id',
            'password'
        ]
        # Configuración de seguridad para la contraseña.
        extra_kwargs = {
            'password': {
                'write_only': True,
                'style': {'input_type': 'password'}
            }
        }

    def create(self, validated_data):
        validated_data['is_active'] = True
        validated_data['is_staff'] = False
        validated_data['is_superuser'] = False
        tipo_user = validated_data.pop('tipousuario', None)
        enfermedad_data = validated_data.pop('enfermedad', None)
        
        # Creamos el usuario usando el managers
        user = Usuario.objects.create_user(**validated_data)
        if tipo_user:
            user.tipousuario = tipo_user
        if enfermedad_data:
            user.enfermedad = enfermedad_data
            
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        
        # 'super()' llama al método de actualización original para los otros campos.
        user = super().update(instance, validated_data)

        if password:
            user.set_password(password)
            user.save()

        return user