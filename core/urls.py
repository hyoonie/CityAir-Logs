from django import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()

#A esto no le muevan bola de nacos
urlpatterns = [
        path('super/', include(router.urls)),

        # Paths: login/home
        path('', views.view_home, name='home'),
        path('sesiones/login', views.view_ct_login, name='login'),

        # Paths: core/template/about us
        path('about_us/aboutus', views.view_aboutus, name='aboutus'),

        # Paths: core/templates/analisis...
        path('analisis/analisis', views.view_ct_analisis, name='analisis'),

        # Paths: core/templates/dispositivos...
        path('dispositivos/dispositivo', views.view_ct_dispositivos, name='dispositivos'),

        # Paths: core/template/reportes...
        path('reportes/reporte', views.view_ct_reportes, name='reportes'),

        # Paths: core/template/alertas...
        path('alertas/alertas', views.view_ct_alertas, name='alertas'),

        ########################
        path('analisis_datos_algoritmos/analisis_datos_home', views.view_home_analisis_datos, name='analisis_datos_home'),

        path('analisis_datos_algoritmos/analisis_datos', views.view_analisis_datos, name='analisis_datos'),

        path('analisis_datos_algoritmos/generador', views.view_generador, name='generador'),

        path('analisis_datos_algoritmos/resumen', views.view_resumen, name='resumen'),

        path('api/kpis/', views.api_get_kpis, name='kpis_api'),
        
        path('analisis_datos_algoritmos/metricas', views.view_metricas, name='metricas'),
    ]