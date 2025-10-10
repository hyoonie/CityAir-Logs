from django.shortcuts import render

# Aqui se definen las vistas de los html dentro de core/templates...

#Home
def view_home(request):
    return render(request, 'home.html')

# core/template/sesiones
def view_ct_login(request):
     return render(request, 'sesiones/login.html')

# core/template/about us
def view_aboutus(request):
        return render(request, 'about_us/about_us.html')

# core/template/datos
def view_ct_analisis(request):
    return render(request, 'analisis/analisis.html')

# core/template/dispositivos
def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

# core/template/reportes
def view_ct_reportes(request):
     return render(request, 'reportes/reportes.html')

# core/template/alertas
def view_ct_alertas(request):
     return render(request, 'alertas/alertas.html')