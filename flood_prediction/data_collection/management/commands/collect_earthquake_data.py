from django.core.management.base import BaseCommand
from data_collection.tasks import fetch_earthquake_data  # Adjust the import path based on your project structure

class Command(BaseCommand):
    help = 'Collect earthquake data and save it to the database'

    def handle(self, *args, **kwargs):
        # Call the fetch_weather_data function
        result = fetch_earthquake_data()

        if "Successfully" in result:
            self.stdout.write(self.style.SUCCESS(result))
        else:
            self.stdout.write(self.style.ERROR(result))
