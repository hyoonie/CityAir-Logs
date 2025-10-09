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
        path('about_us/aboutus', views.view_aboutus, name='aboutus'),

        # Paths: core/templates/datos...
        path('datos/dato1', views.view_ct_datos, name='ct_datos1'),

        # Paths: core/templates/dispositivos...
        path('dispositivos/dispositivo', views.view_ct_dispositivos, name='dispositivos'),

        # Paths: core/template/reportes...
        path('reportes/reporte', views.view_ct_reportes, name='reportes'),
    ]