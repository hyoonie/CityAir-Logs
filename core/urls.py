from django import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from . import views_analisis_datos as vad 
router = DefaultRouter()

#A esto no le muevan bola de nacos
urlpatterns = [
    path('super/', include(router.urls)),

    # ───── HOME / LOGIN ─────
    path('', views.view_home, name='home'),
    path('sesiones/login', views.view_ct_login, name='login'),

    # ───── ABOUT US ─────
    path('about_us/aboutus', views.view_aboutus, name='aboutus'),

    # ───── ANALISIS / DISPOSITIVOS / REPORTES / ALERTAS ─────
    path('analisis/analisis', views.view_ct_analisis, name='analisis'),
    path('dispositivos/dispositivo', views.view_ct_dispositivos, name='dispositivos'),
    path('reportes/reporte', views.view_ct_reportes, name='reportes'),
    path('alertas/alertas', views.view_ct_alertas, name='alertas'),

    # ───── ANALISIS DE DATOS ─────
    path('analisis_rl_rf/analisis_home', vad.view_ct_analisis_home, name='analisis_home'),
    path('analisis_datos_algoritmos/analisis_datos', vad.view_ct_analisis, name='analisis_datos'),
    path('analisis_datos_algoritmos/generador', vad.view_ct_generador, name='generador'),
    path('analisis_datos_algoritmos/resumen', vad.view_ct_resumen, name='resumen'),
    path('analisis_datos_algoritmos/metricas', vad.view_ct_metricas, name='metricas'),

    # Mongo y CSV interactivo
    path('analisis_datos_mongodb/analisis_mongo', vad.view_ct_analisis_mongo, name='analisis_mongo'),
    path('analisis_rl_rf/analisis_csv', vad.view_ct_analisis_csv, name='analisis_csv'),

    # ───── API ─────
    path('api/kpis/', vad.api_get_kpis, name='kpis_api'),
]