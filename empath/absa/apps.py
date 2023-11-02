from django.apps import AppConfig


class AbsaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'absa'

    def ready(self) -> None:
        import absa.signals
