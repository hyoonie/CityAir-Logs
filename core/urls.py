from django import views
from django.urls import path, include
from django.contrib import admin
from rest_framework.routers import DefaultRouter
from . import views
from .views import SensorDataUploadView

router = DefaultRouter()

#A esto no le muevan bola de nacos
urlpatterns = [
        path('admin/', admin.site.urls),

        # Paths: sesiones/login-home-logout
        path('', views.view_home, name='home'),
        path('login/', views.view_ct_login, name='login'),
        path('logout/', views.view_ct_logout, name='logout'),

        # Paths: sesiones/cuenta
        path('perfil/<int:user_id>', views.view_profile, name='configuracion'),
        path('perfil/<int:user_id>/alertas', views.view_ct_ca_alertas, name='alertas'),

        # Paths: sesieones/registros
        path('sesiones/registro', views.view_cts_register, name='registro'),
        #path('sesiones/registro', views.registro_usuarios, name='registro'),
        path('api/registro_usuarios/', views.registro_usuarios, name='registrar'),

        # Paths: core/template/about us
        path('sobre-nosotros', views.view_aboutus, name='sobre-nosotros'),

        # Paths: core/template/legal
        path('terminos-de-uso', views.view_terminosuso, name='terminos-de-uso'),
        path('politicas-de-privacidad', views.view_politicaprivacidad, name='politicas-de-privacidad'),

        # Paths: core/templates/datos...
        path('analisis-datos', views.view_ct_datos, name='analisis-de-datos'),

        # Paths: core/templates/dispositivos...
        path('mapa-de-dispositivos', views.view_ct_dispositivos, name='dispositivos'),

        # Paths: core/template/reportes...
        path('dato-puro', views.view_ct_reportes, name='reportes'),

        # Paths core/template/analisis_predictivo
        #path('analisis_predictivo/resultados', views.view_ct_resultados_modelos, name='resultado-analisis-predictivo'),
        path('analisis-predictivo', views.view_ct_tabla_predicciones, name='analisis-predictivo'),


        # API dispositivos
        path('api/v1/sensor/upload/', SensorDataUploadView.as_view(), name='sensor-upload'),
    ]
