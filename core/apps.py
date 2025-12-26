from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        from django.db.backends.base.operations import BaseDatabaseOperations
        if 'ObjectIdField' not in BaseDatabaseOperations.integer_field_ranges:
            BaseDatabaseOperations.integer_field_ranges['ObjectIdField'] = (-9223372036854775808, 9223372036854775807)