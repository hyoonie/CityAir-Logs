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

    # ───── ANALISIS DE DATOS  GENERAL─────
    path('analisis_rl_rf/analisis_home', vad.view_ct_analisis_home, name='analisis_home'),
    # ───── Imagenes─────
    path('analisis_rl_rf/resultados_modelos', vad.view_ct_resultados_modelos, name='resultados_modelos'),
    # ───── Tabla de datos─────
    path('analisis_rl_rf/tabla_predicciones', vad.view_ct_tabla_predicciones, name='tabla_predicciones'),
]