
import logging
from django.utils import timezone
from data_collection.models import RiverData, WeatherData
from .combined_lstm_model import train_combined_lstm_model, save_combined_model_and_scaler
from .lstm_model import train_lstm_model, save_model_and_scaler
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def train_models():
    logger.info("Starting model training...")
    
    # Fetch data
    river_data = RiverData.objects.all().order_by('date')
    weather_data = WeatherData.objects.all().order_by('date')
    
    if not river_data or not weather_data:
        logger.warning("No data available for training.")
        return
    
    # Prepare data for LSTM model
    river_df = pd.DataFrame(list(river_data.values('date', 'river_discharge')))
    river_discharge_data = river_df['river_discharge'].values
    
    # Train LSTM model
    try:
        lstm_model, lstm_scaler = train_lstm_model(river_discharge_data)
        save_model_and_scaler(lstm_model, lstm_scaler, 'lstm_model.h5', 'lstm_scaler.pkl')
        logger.info("LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
    
    # Prepare data for combined LSTM model
    weather_df = pd.DataFrame(list(weather_data.values('date', 'temperature', 'humidity', 'pressure')))
    combined_df = pd.merge(river_df, weather_df, on='date', how='inner')
    
    # Train combined LSTM model
    try:
        combined_model, combined_scaler = train_combined_lstm_model(river_df, weather_df)
        save_combined_model_and_scaler(combined_model, combined_scaler, 'combined_lstm_model.h5', 'combined_lstm_scaler.pkl')
        logger.info("Combined LSTM model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training combined LSTM model: {str(e)}")
    
    logger.info("Model training completed.")