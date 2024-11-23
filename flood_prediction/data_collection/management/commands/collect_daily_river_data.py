from django.core.management.base import BaseCommand
from data_collection.tasks import fetch_all_data, fetch_river_data, fetch_weather_data

class Command(BaseCommand):
    help = 'Collect data (river discharge and/or weather data) from 2022-01-01 to the current date.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-type',
            type=str,
            choices=['river', 'weather', 'all'],
            default='all',
            help='Specify the type of data to fetch: "river", "weather", or "all". Default is "all".'
        )

    def handle(self, *args, **options):
        data_type = options['data_type']

        if data_type == 'river':
            fetch_river_data()
            self.stdout.write(self.style.SUCCESS('Successfully fetched river discharge data.'))
        elif data_type == 'weather':
            fetch_weather_data()
            self.stdout.write(self.style.SUCCESS('Successfully fetched weather data.'))
        elif data_type == 'all':
            fetch_all_data()  # Use the combined function for 'all'
            self.stdout.write(self.style.SUCCESS('Successfully fetched all data (river discharge and weather).'))
        else:
            self.stdout.write(self.style.ERROR('Invalid data type specified.'))
