from django import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()

#A esto no le muevan bola de nacos
urlpatterns = [
        path('super/', include(router.urls)),

        # Paths: sesiones/login/home
        path('', views.view_home, name='home'),
        path('sesiones/login', views.view_ct_login, name='login'),

        # Paths: sesiones/cuenta
        path('profile/<int:user_id>/', views.view_profile, name='cuenta'),

        # Paths: sesieones/registros
        path('sesiones/registro', views.view_cts_register, name='registro'),
        #path('sesiones/registro', views.registro_usuarios, name='registro'),
        path('api/registro_usuarios/', views.registro_usuarios, name='registrar'),

        # Paths: core/template/about us
        path('sobre-nosotros', views.view_aboutus, name='aboutus'),

        # Paths: core/templates/datos...
        path('analisis_datos', views.view_ct_datos, name='analisis_datos'),

        # Paths: core/templates/dispositivos...
        path('dispositivos', views.view_ct_dispositivos, name='dispositivos'),

        # Paths: core/template/reportes...
        path('reportes', views.view_ct_reportes, name='reportes'),

        # Paths core/template/analisis_predictivo
        #path('analisis_predictivo/resultados', views.view_ct_resultados_modelos, name='resultado-analisis-predictivo'),
        path('analisis_predictivo/predicciones', views.view_ct_tabla_predicciones, name='predicciones'),

    ]