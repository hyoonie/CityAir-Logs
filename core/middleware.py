from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect
from django.urls import reverse


class ForzarCambioPasswordMiddleware:
    """Mientras el usuario tenga una contraseña temporal, solo puede ir a cambiarla o cerrar sesión.
    Se registra en MIDDLEWARE después de AuthenticationMiddleware y MessageMiddleware."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, 'user', None)
        if user is not None and user.is_authenticated and getattr(user, 'debe_cambiar_password', False):
            permitidas = (
                reverse('configuracion', args=[user.id]),
                reverse('cambiar-password'),
                reverse('logout'),
            )
            if request.path not in permitidas and not request.path.startswith(settings.STATIC_URL):
                messages.warning(request, 'Debes cambiar tu contraseña temporal antes de continuar.')
                return redirect('configuracion', user_id=user.id)
        return self.get_response(request)
