import os
from django.http import JsonResponse
from model_trainer.lstm_model import load_model_and_scaler, make_prediction
from model_trainer.combined_lstm_model import load_combined_model_and_scaler, make_combined_prediction
from data_collection.models import RiverData, WeatherData
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def predict_7_days(request):
    model_path = 'latest_model.keras'
    scaler_path = 'latest_scaler.joblib'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return JsonResponse({'error': 'Model or scaler not found. Please train the model first.'}, status=400)

    model, scaler = load_model_and_scaler(model_path, scaler_path)
    latest_data = list(RiverData.objects.all().order_by('-date')[:7].values_list('river_discharge', flat=True))
    
    if len(latest_data) < 7:
        return JsonResponse({'error': 'Not enough data for prediction. Need at least 7 data points.'}, status=400)

    predictions = []
    dates = []
    input_data = np.array(latest_data[::-1])  
    # input_data = np.array(latest_data).reshape(1, -1, 1)  # Shape (1, 7, 1)

    last_date = RiverData.objects.latest('date').date

    for i in range(7):
        prediction = make_prediction(model, scaler, input_data)
        predictions.append(float(prediction))
        input_data = np.roll(input_data, -1)
        input_data[-1] = prediction
        
        prediction_date = last_date + timedelta(days=i+1)
        dates.append(prediction_date.strftime('%Y-%m-%d'))

    return JsonResponse({'predictions': predictions, 'dates': dates})

def predict_7_days_combined(request):
    model_path = 'latest_combined_model.keras'
    scaler_path = 'latest_combined_scaler.joblib'
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return JsonResponse({'error': 'Combined model or scaler not found. Please train the combined model first.'}, status=400)

    model, scaler = load_combined_model_and_scaler(model_path, scaler_path)

    latest_river_data = RiverData.objects.all().order_by('-date')[:7]
    latest_weather_data = WeatherData.objects.all().order_by('-date')[:7]

    if len(latest_river_data) < 7 or len(latest_weather_data) < 7:
        return JsonResponse({'error': 'Not enough data for prediction. Need at least 7 days of both river and weather data.'}, status=400)

    combined_data = pd.DataFrame({
        'date': [data.date for data in latest_river_data],
        'river_discharge': [data.river_discharge for data in latest_river_data],
        'temperature': [data.temperature for data in latest_weather_data],
        'humidity': [data.humidity for data in latest_weather_data],
        'pressure': [data.pressure for data in latest_weather_data]
    }).sort_values('date')

    predictions = []
    dates = []
    input_data = combined_data[['river_discharge', 'temperature', 'humidity', 'pressure']].values
    # input_data = combined_data[['river_discharge', 'temperature', 'humidity', 'pressure']].values.reshape(1, -1, 4)  # Shape (1, 7, 4)

    last_date = combined_data['date'].max()

    for i in range(7):
        prediction = make_combined_prediction(model, scaler, input_data)
        predictions.append(float(prediction))

        new_row = input_data[-1].copy()
        new_row[0] = prediction  
        input_data = np.roll(input_data, -1, axis=0)
        input_data[-1] = new_row

        prediction_date = last_date + timedelta(days=i+1)
        dates.append(prediction_date.strftime('%Y-%m-%d'))

    return JsonResponse({'predictions': predictions, 'dates': dates})
   

def predict(request):
    model_path = 'latest_model.keras'
    scaler_path = 'latest_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.error(f"Model or scaler not found. Model path: {model_path}, Scaler path: {scaler_path}")
        return JsonResponse({'error': 'Model or scaler not found. Please train the model first.'}, status=400)

    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)

    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        return JsonResponse({'error': 'Error loading model or scaler.'}, status=500)

    try:
        current_discharge = float(request.GET.get('current_discharge', 0))
    except ValueError:
        logger.error(f"Invalid input: {request.GET.get('current_discharge')}")
        return JsonResponse({'error': 'Invalid input. Please provide a valid number.'}, status=400)

    input_sequence = np.array([current_discharge] * 7).reshape(-1, 1)
    
    try:
        scaled_input = scaler.transform(input_sequence).reshape(1, 7, 1)  # Shape (1, 7, 1)
    except Exception as e:
        logger.error(f"Error scaling input: {str(e)}")
        return JsonResponse({'error': 'Error scaling input.'}, status=500)

    try:
        predictions = []
        for _ in range(7):
            scaled_prediction = model.predict(scaled_input)[0][0]
            prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]
            predictions.append(float(prediction))

            scaled_input = np.roll(scaled_input, -1, axis=1)
            scaled_input[0, -1, 0] = scaled_prediction

        return JsonResponse({'predictions': predictions})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JsonResponse({'error': 'Error during prediction.'}, status=500)


def predict_combined(request):
    model_path = 'latest_combined_model.keras'
    scaler_path = 'latest_combined_scaler.joblib'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return JsonResponse({'error': 'Combined model or scaler not found. Please train the combined model first.'}, status=400)

    model, scaler = load_combined_model_and_scaler(model_path, scaler_path)

    try:
        river_discharge = float(request.GET.get('river_discharge', 0))
        temperature = float(request.GET.get('temperature', 0))
        humidity = float(request.GET.get('humidity', 0))
        pressure = float(request.GET.get('pressure', 0))
    except ValueError:
        return JsonResponse({'error': 'Invalid input. Please provide valid numbers for all parameters.'}, status=400)

    input_sequence = np.array([
        [river_discharge, temperature, humidity, pressure]
    ] * 7).reshape(1, 7, 4)  

    scaled_input = scaler.transform(input_sequence.reshape(-1, 4)).reshape(1, 7, 4)

    predictions = []
    for _ in range(7):
        scaled_prediction = model.predict(scaled_input)[0][0]  
        prediction = scaler.inverse_transform([[scaled_prediction] + [temperature, humidity, pressure]])[0][0]
        predictions.append(float(prediction))

        scaled_input = np.roll(scaled_input, -1, axis=1)
        scaled_input[0, -1, 0] = scaled_prediction

    return JsonResponse({'predictions': predictions})

