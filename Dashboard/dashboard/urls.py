from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    
    # NUEVA RUTA para la página de Análisis y Gráficas
    path('analisis/', views.analisis_view, name='analisis_data'), 
    path('quienes/', views.quienes, name='quienes'),
    path('resumen/', views.resumen, name='resumen'),
    path('generador/', views.generador, name='generador'),
    path('api/kpis/', views.get_kpis_api, name='kpis_api'), 
]