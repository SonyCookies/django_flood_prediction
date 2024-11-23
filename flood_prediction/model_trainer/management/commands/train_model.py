from django.core.management.base import BaseCommand
from data_collection.models import RiverData
from model_trainer.lstm_model import train_lstm_model, save_model_and_scaler
import numpy as np

class Command(BaseCommand):
    help = 'Train LSTM model on latest data'

    def handle(self, *args, **options):
        data = RiverData.objects.all().order_by('date').values_list('river_discharge', flat=True)
        data = np.array(data)
        
        if len(data) < 8:
            self.stdout.write(self.style.WARNING('Not enough data to train the model. Need at least 8 data points.'))
            return

        try:
            model, scaler = train_lstm_model(data)
            save_model_and_scaler(model, scaler, 'latest_model.keras', 'latest_scaler.joblib')
            self.stdout.write(self.style.SUCCESS('Successfully trained and saved model and scaler'))
        except ValueError as e:
            self.stdout.write(self.style.ERROR(f'Error training model: {str(e)}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Unexpected error: {str(e)}'))