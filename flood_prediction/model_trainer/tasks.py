
import logging
from django.utils import timezone
from data_collection.models import RiverData, WeatherData, EarthquakeData
from .combined_lstm_model import train_combined_lstm_model, save_combined_model_and_scaler
from .lstm_model import train_lstm_model, save_model_and_scaler
from .earthquake_lstm_model import train_earthquake_model, save_model_and_scaler as save_earthquake_model_and_scaler
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def train_models():
    logger.info("Starting model training...")
    
    river_data = RiverData.objects.all().order_by('date')
    weather_data = WeatherData.objects.all().order_by('date')
    earthquake_data = EarthquakeData.objects.all().order_by('date')
    
    if not river_data or not weather_data or not earthquake_data:
        logger.warning("No data available for training.")
        return
    
    # Train River Discharge LSTM model
    river_df = pd.DataFrame(list(river_data.values('date', 'river_discharge')))
    river_discharge_data = river_df['river_discharge'].values
    
    try:
        lstm_model, lstm_scaler = train_lstm_model(river_discharge_data)
        save_model_and_scaler(lstm_model, lstm_scaler, 'lstm_model.h5', 'lstm_scaler.pkl')
        logger.info("River Discharge LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training River Discharge LSTM model: {str(e)}")
    
    # Train Earthquake LSTM model
    try:
        earthquake_model, earthquake_scaler = train_earthquake_model(earthquake_data)
        save_earthquake_model_and_scaler(earthquake_model, earthquake_scaler, 'earthquake_lstm_model.h5', 'earthquake_scaler.pkl')
        logger.info("Earthquake LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training Earthquake LSTM model: {str(e)}")
    
    # Prepare data for combined LSTM model
    weather_df = pd.DataFrame(list(weather_data.values('date', 'temperature', 'humidity', 'pressure')))
    earthquake_df = pd.DataFrame(list(earthquake_data.values('date', 'magnitude')))
    combined_df = pd.merge(river_df, weather_df, on='date', how='inner')
    combined_df = pd.merge(combined_df, earthquake_df, on='date', how='outer')
    combined_df = combined_df.sort_values('date').fillna(method='ffill')
    
    # Train combined LSTM model
    try:
        combined_model, combined_scaler = train_combined_lstm_model(combined_df)
        save_combined_model_and_scaler(combined_model, combined_scaler, 'combined_lstm_model.h5', 'combined_lstm_scaler.pkl')
        logger.info("Combined LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training combined LSTM model: {str(e)}")
    
    logger.info("Model training completed.")
