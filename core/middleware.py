from django.conf import settings
from django.shortcuts import redirect
from django.urls import resolve, Resolver404


class CambioPasswordObligatorioMiddleware:
    # Mientras debe_cambiar_password sea True, el usuario solo puede ver su PROPIO perfil,
    # enviar el cambio de contraseña y cerrar sesión. Todo lo demás lo redirige a su perfil.
    PERMITIDAS = {'cambiar-password', 'logout'}

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = request.user
        if user.is_authenticated and user.debe_cambiar_password:
            if not self._permitida(request, user):
                return redirect('configuracion', user_id=user.id)
        return self.get_response(request)

    def _permitida(self, request, user):
        # Los estáticos deben cargar, o la página de cambio se vería rota
        for prefijo in (settings.STATIC_URL, getattr(settings, 'MEDIA_URL', None)):
            if prefijo and prefijo != '/' and request.path_info.startswith(prefijo):
                return True
        try:
            match = resolve(request.path_info)
        except Resolver404:
            return False
        if match.url_name in self.PERMITIDAS:
            return True
        # El perfil solo se permite si es el PROPIO
        return match.url_name == 'configuracion' and match.kwargs.get('user_id') == user.id