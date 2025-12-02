from django.shortcuts import render

def view_home(request):
    return render(request, 'home.html')

def view_ct_login(request):
    return render(request, 'sesiones/login.html')

def view_aboutus(request):
    return render(request, 'about_us/about_us.html')

def view_ct_analisis(request):
    return render(request, 'analisis/analisis.html')

def view_ct_dispositivos(request):
    return render(request, 'dispositivos/dispositivos.html')

def view_ct_reportes(request):
    return render(request, 'reportes/reportes.html')

def view_ct_alertas(request):
    return render(request, 'alertas/alertas.html')

