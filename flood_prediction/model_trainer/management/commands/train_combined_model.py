# management/commands/train_combined_model.py

from django.core.management.base import BaseCommand
from data_collection.models import RiverData, WeatherData
from model_trainer.combined_lstm_model import train_combined_lstm_model, save_combined_model_and_scaler
import pandas as pd

class Command(BaseCommand):
    help = 'Train combined LSTM model on latest river and weather data'

    def handle(self, *args, **options):
        river_data = pd.DataFrame(RiverData.objects.all().order_by('date').values('date', 'river_discharge'))
        weather_data = pd.DataFrame(WeatherData.objects.all().order_by('date').values('date', 'temperature', 'humidity', 'pressure'))
        
        if len(river_data) < 8 or len(weather_data) < 8:
            self.stdout.write(self.style.WARNING('Not enough data to train the model. Need at least 8 data points for both river and weather data.'))
            return

        try:
            model, scaler = train_combined_lstm_model(river_data, weather_data)
            save_combined_model_and_scaler(model, scaler, 'latest_combined_model.keras', 'latest_combined_scaler.joblib')
            self.stdout.write(self.style.SUCCESS('Successfully trained and saved combined model and scaler'))
        except ValueError as e:
            self.stdout.write(self.style.ERROR(f'Error training model: {str(e)}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Unexpected error: {str(e)}'))