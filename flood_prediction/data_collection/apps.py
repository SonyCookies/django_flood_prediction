from django.apps import AppConfig
from django.conf import settings

class DataCollectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'data_collection'

    def ready(self):
        if settings.DEBUG:
            from .tasks import start_scheduler
            import os
            if os.environ.get('RUN_MAIN', None) != 'true':
                start_scheduler()