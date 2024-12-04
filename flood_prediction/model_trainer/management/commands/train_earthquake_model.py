from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from data_collection.models import EarthquakeData
from model_trainer.earthquake_lstm_model import (
    train_earthquake_model, 
    save_model_and_scaler, 
    predict_earthquake, 
    predict_next_30_days, 
    plot_predictions,
    load_model_and_scaler
)
import numpy as np
import os

class Command(BaseCommand):
    help = 'Train LSTM model on latest earthquake data, make predictions, and generate 30-day forecast'

    def add_arguments(self, parser):
        parser.add_argument('--predict_date', type=str, help='Date to predict earthquake probability (YYYY-MM-DD)')
        parser.add_argument('--forecast', action='store_true', help='Generate 30-day forecast')

    def handle(self, *args, **options):
        end_date = timezone.now()
        start_date = end_date - timedelta(days=30)
        earthquakes = list(EarthquakeData.objects.filter(date__range=(start_date, end_date)).order_by('date').values('date', 'magnitude', 'depth', 'latitude', 'longitude'))

        if len(earthquakes) < 8:
            self.stdout.write(self.style.WARNING('Not enough data to train the model. Need at least 8 data points.'))
            return

        try:
            model_path = 'earthquake_lstm_model.keras'
            scaler_path = 'earthquake_scaler.joblib'

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                self.stdout.write("Training new model...")
                model, scaler = train_earthquake_model(earthquakes)
                self.stdout.write("Saving the model and scaler...")
                save_model_and_scaler(model, scaler, model_path, scaler_path)
            else:
                self.stdout.write("Loading existing model...")
                model, scaler = load_model_and_scaler(model_path, scaler_path)

            last_week_data = np.array([[eq['magnitude'], eq['depth'], eq['latitude'], eq['longitude']] for eq in earthquakes[-7:]])
            historical_data = np.array([[eq['magnitude'], eq['depth'], eq['latitude'], eq['longitude']] for eq in earthquakes])

            if options['predict_date']:
                target_date = timezone.make_aware(datetime.strptime(options['predict_date'], '%Y-%m-%d'))
                self.stdout.write(f"Making prediction for {target_date.date()}...")
                prediction = predict_earthquake(model, scaler, last_week_data, historical_data, target_date)

                self.stdout.write(f"Earthquake prediction for {target_date.date()}:")
                self.stdout.write(f"Earthquake probability: {prediction['earthquake_probability']:.2%}")
                self.stdout.write(f"Magnitude: {prediction['magnitude']:.2f}")
                self.stdout.write(f"Depth: {prediction['depth']:.2f}")
                self.stdout.write(f"Latitude: {prediction['latitude']:.5f}")
                self.stdout.write(f"Longitude: {prediction['longitude']:.5f}")

            if options['forecast']:
                self.stdout.write("Generating 30-day forecast...")
                start_date = timezone.now()
                predictions = predict_next_30_days(model, scaler, last_week_data, historical_data, start_date)
                plot_predictions(predictions)
                self.stdout.write("30-day forecast plot saved as 'earthquake_prediction_30days.png'")

                # Print the first day's prediction as an example
                first_day_prediction = predictions[0]
                self.stdout.write(f"\nFirst day prediction ({first_day_prediction['date'].date()}):")
                self.stdout.write(f"Earthquake probability: {first_day_prediction['earthquake_probability']:.2%}")
                self.stdout.write(f"Magnitude: {first_day_prediction['magnitude']:.2f}")
                self.stdout.write(f"Depth: {first_day_prediction['depth']:.2f}")
                self.stdout.write(f"Latitude: {first_day_prediction['latitude']:.5f}")
                self.stdout.write(f"Longitude: {first_day_prediction['longitude']:.5f}")

            self.stdout.write(self.style.SUCCESS("Model training, prediction, and forecasting completed successfully"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred: {str(e)}"))

