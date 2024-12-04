from datetime import datetime, timedelta
from django.utils.timezone import now
from django.shortcuts import render
from data_collection.models import RiverData, WeatherData
from prediction.views import predict_7_days, predict_7_days_combined
from django.http import JsonResponse
import json
import numpy as np
from django.db.models import Avg, Max, Min
from statistics import median, mode
from django.db.models.functions import TruncDate
from django.utils import timezone
from statistics import mean, median, mode, stdev, variance
from scipy import stats



def calculate_risk_level(discharge):
    if discharge < 2:
        return "Low"
    elif 2 <= discharge < 20:
        return "Moderate"
    else:
        return "High"

def dashboard(request):
    historical_data = RiverData.objects.filter(
        date__gte=now() - timedelta(days=7)
    ).order_by('-date')[:7]
    
    historical_data = reversed(historical_data)

    latest_data = RiverData.objects.order_by('-date').first()
    latest_weather = WeatherData.objects.order_by('-date').first()

    forecast_response = predict_7_days(request)
    combined_forecast_response = predict_7_days_combined(request)
    forecast_data = None
    combined_forecast_data = None

    if isinstance(forecast_response, JsonResponse):
        forecast_content = json.loads(forecast_response.content.decode('utf-8'))
        if 'error' not in forecast_content:
            forecast_data = [
                {
                    "day": (datetime.now() + timedelta(days=i + 1)).strftime('%A'),
                    "date": d,
                    "value": v,
                    "risk": calculate_risk_level(v)
                }
                for i, (d, v) in enumerate(zip(forecast_content['dates'], forecast_content['predictions']))
            ]

    if isinstance(combined_forecast_response, JsonResponse):
        combined_forecast_content = json.loads(combined_forecast_response.content.decode('utf-8'))
        if 'error' not in combined_forecast_content:
            combined_forecast_data = [
                {
                    "day": (datetime.now() + timedelta(days=i + 1)).strftime('%A'),
                    "date": d,
                    "value": v,
                    "risk": calculate_risk_level(v)
                }
                for i, (d, v) in enumerate(zip(combined_forecast_content['dates'], combined_forecast_content['predictions']))
            ]

    historical_data_serialized = [
        {"date": data.date.strftime('%Y-%m-%d'), "level": data.river_discharge} for data in historical_data
    ]

    return render(request, 'dashboard.html', {
        'forecast_data': forecast_data or [], 
        'combined_forecast_data': combined_forecast_data or [],
        'historical_data': historical_data_serialized, 
        'historical_data_count': len(historical_data_serialized), 
        'current_river_discharge': {
            "date": latest_data.date.strftime('%Y-%m-%d'),
            "level": latest_data.river_discharge
        } if latest_data else None,
        'current_weather': {
            "temperature": latest_weather.temperature,
            "humidity": latest_weather.humidity,
            "pressure": latest_weather.pressure
        } if latest_weather else None,
        'error_message': forecast_content.get('error') if 'error' in forecast_content else None,
    })

def welcome(request):
    return render(request, 'welcome.html')

def predict_flood(request):
    return render(request, 'predict.html')

def get_history(request):
    try:
        all_river_data = RiverData.objects.all()
        all_weather_data = WeatherData.objects.all()

        thirty_days_ago = timezone.now().date() - timedelta(days=30)

        recent_river_data = all_river_data.filter(date__gte=thirty_days_ago).order_by('date')
        recent_weather_data = all_weather_data.filter(date__gte=thirty_days_ago).order_by('date')

        def calculate_stats(values):
            values = [v for v in values if v is not None]
            if not values:
                return {'mean': 0, 'median': 0, 'mode': 0, 'max': 0, 'min': 0, 'std_dev': 0, 'variance': 0, 'range': 0, 'iqr': 0}
            
            sorted_values = sorted(values)
            q1, q3 = np.percentile(sorted_values, [25, 75])
            
            return {
                'mean': round(mean(values), 2),
                'median': round(median(values), 2),
                'mode': round(mode(values), 2),
                'max': round(max(values), 2),
                'min': round(min(values), 2),
                'std_dev': round(stdev(values), 2),
                'variance': round(variance(values), 2),
                'range': round(max(values) - min(values), 2),
                'iqr': round(q3 - q1, 2),
            }

        stats = {
            'river_discharge': calculate_stats(all_river_data.values_list('river_discharge', flat=True)),
            'temperature': calculate_stats(all_weather_data.values_list('temperature', flat=True)),
            'humidity': calculate_stats(all_weather_data.values_list('humidity', flat=True)),
            'pressure': calculate_stats(all_weather_data.values_list('pressure', flat=True)),
        }
        
        chart_data = []
        for date in (timezone.now().date() - timedelta(days=x) for x in range(29, -1, -1)):
            river = recent_river_data.filter(date=date).first()
            weather = recent_weather_data.filter(date=date).first()
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'river_discharge': round(river.river_discharge, 2) if river and river.river_discharge is not None else None,
                'temperature': round(weather.temperature, 2) if weather and weather.temperature is not None else None,
                'humidity': round(weather.humidity, 2) if weather and weather.humidity is not None else None,
                'pressure': round(weather.pressure, 2) if weather and weather.pressure is not None else None,
            })

        def calculate_moving_average(data, window):
            return [round(sum(data[max(i-window+1, 0):i+1]) / min(i+1, window), 2) for i in range(len(data))]

        def calculate_linear_trend(data):
            x = list(range(len(data)))
            y = data
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            return [round(m * xi + c, 2) for xi in x]

        def calculate_seasonal_patterns(data):
            seasonal_data = [0] * 12
            counts = [0] * 12
            for i, value in enumerate(data):
                month = i % 12
                seasonal_data[month] += value
                counts[month] += 1
            return [round(seasonal_data[i] / counts[i], 2) if counts[i] > 0 else 0 for i in range(12)]

        def calculate_ema(data, period):
            ema = [data[0]]
            k = 2 / (period + 1)
            for i in range(1, len(data)):
                ema.append(round(data[i] * k + ema[-1] * (1 - k), 2))
            return ema

        trend_analysis = {}
        for metric in ['river_discharge', 'temperature', 'humidity', 'pressure']:
            values = [item[metric] for item in chart_data if item[metric] is not None]
            if values:
                trend_analysis[metric] = {
                    'moving_average_7': calculate_moving_average(values, 7),
                    'moving_average_30': calculate_moving_average(values, 30),
                    'linear_trend': calculate_linear_trend(values),
                    'seasonal_patterns': calculate_seasonal_patterns(values),
                    'ema': calculate_ema(values, 7)
                }
            else:
                trend_analysis[metric] = {
                    'moving_average_7': [],
                    'moving_average_30': [],
                    'linear_trend': [],
                    'seasonal_patterns': [],
                    'ema': []
                }

        return JsonResponse({
            'stats': stats,
            'chart_data': chart_data,
            'trend_analysis': trend_analysis
        })

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JsonResponse({'error': 'An error occurred while processing historical data.'}, status=500)
    

def history(request):
    return render(request, 'historical_data.html')

def earthquake(request):
    return render(request, 'earthquake_prediction.html')


def get_correlation_data(request):
    try:
        all_river_data = RiverData.objects.all()
        all_weather_data = WeatherData.objects.all()

        common_dates = set(all_river_data.values_list('date', flat=True)) & set(all_weather_data.values_list('date', flat=True))
        
        data = {
            'river_discharge': [],
            'temperature': [],
            'humidity': [],
            'pressure': []
        }

        for date in common_dates:
            river_data = all_river_data.filter(date=date).first()
            weather_data = all_weather_data.filter(date=date).first()
            
            if river_data and weather_data:
                data['river_discharge'].append(river_data.river_discharge)
                data['temperature'].append(weather_data.temperature)
                data['humidity'].append(weather_data.humidity)
                data['pressure'].append(weather_data.pressure)

        correlations = {
            'temperature': stats.pearsonr(data['river_discharge'], data['temperature'])[0],
            'humidity': stats.pearsonr(data['river_discharge'], data['humidity'])[0],
            'pressure': stats.pearsonr(data['river_discharge'], data['pressure'])[0]
        }

        scatter_data = {
            'temperature': list(zip(data['temperature'], data['river_discharge'])),
            'humidity': list(zip(data['humidity'], data['river_discharge'])),
            'pressure': list(zip(data['pressure'], data['river_discharge']))
        }

        return JsonResponse({
            'correlations': correlations,
            'scatter_data': scatter_data
        })

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JsonResponse({'error': 'An error occurred while processing correlation data.'}, status=500)

def correlation_relationship(request):
    return render(request, 'correlation_relationship.html')

def get_trend_data(request):
    """
    Retrieve and process data for line charts showing trends over time.
    Limited to the last 100 days.
    """
    try:
        # Calculate the date 100 days ago
        cutoff_date = datetime.now() - timedelta(days=100)

        # Filter data for the last 100 days
        river_data = RiverData.objects.filter(date__gte=cutoff_date).order_by('date')
        weather_data = WeatherData.objects.filter(date__gte=cutoff_date).order_by('date')

        # Extract data
        labels = [data.date.strftime('%Y-%m-%d') for data in river_data]
        river_discharge = [data.river_discharge for data in river_data]

        # Match weather data with river data
        temperature = [
            weather_data.filter(date=data.date).first().temperature
            for data in river_data if weather_data.filter(date=data.date).exists()
        ]
        humidity = [
            weather_data.filter(date=data.date).first().humidity
            for data in river_data if weather_data.filter(date=data.date).exists()
        ]
        pressure = [
            weather_data.filter(date=data.date).first().pressure
            for data in river_data if weather_data.filter(date=data.date).exists()
        ]

        # Return response
        return JsonResponse({
            'labels': labels,
            'data': {
                'river_discharge': river_discharge,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
            },
        })

    except Exception as e:
        print(f"Error in get_trend_data: {str(e)}")
        return JsonResponse({'error': 'An error occurred while processing trend data.'}, status=500)
    
def get_category_data(request):
    """
    Retrieve and process category data (e.g., classifications of river discharge,
    temperature, humidity, pressure, etc.) for use in charts or visualizations.
    """
    try:
        # Fetch all river and weather data
        river_data = RiverData.objects.all().order_by('date')
        weather_data = WeatherData.objects.all().order_by('date')

        # Example of categories: Group river discharge values into categories (Low, Moderate, High)
        river_discharge_categories = categorize_river_discharge(river_data)

        # Example of categories for weather data: Classify temperature, humidity, and pressure
        temperature_categories = categorize_weather_data(weather_data, 'temperature')
        humidity_categories = categorize_weather_data(weather_data, 'humidity')
        pressure_categories = categorize_weather_data(weather_data, 'pressure')

        # Return the categories data in the response
        return JsonResponse({
            'river_discharge_categories': river_discharge_categories,
            'temperature_categories': temperature_categories,
            'humidity_categories': humidity_categories,
            'pressure_categories': pressure_categories,
        })

    except Exception as e:
        print(f"Error in get_category_data: {str(e)}")
        return JsonResponse({'error': 'An error occurred while processing category data.'}, status=500)

def categorize_river_discharge(river_data):
    """
    Categorize river discharge values into categories (e.g., Low, Moderate, High).
    """
    categories = {'Low': 0, 'Moderate': 0, 'High': 0}
    
    for data in river_data:
        if data.river_discharge < 5:
            categories['Low'] += 1
        elif 5 <= data.river_discharge < 10:
            categories['Moderate'] += 1
        else:
            categories['High'] += 1
    
    return categories

def categorize_weather_data(weather_data, category_type):
    """
    Categorize weather data (temperature, humidity, or pressure) into ranges.
    """
    categories = {'Low': 0, 'Moderate': 0, 'High': 0}
    
    for data in weather_data:
        value = getattr(data, category_type)
        
        if category_type == 'temperature':
            if value < 10:
                categories['Low'] += 1
            elif 10 <= value < 25:
                categories['Moderate'] += 1
            else:
                categories['High'] += 1
        elif category_type == 'humidity':
            if value < 30:
                categories['Low'] += 1
            elif 30 <= value < 60:
                categories['Moderate'] += 1
            else:
                categories['High'] += 1
        elif category_type == 'pressure':
            if value < 1010:
                categories['Low'] += 1
            elif 1010 <= value < 1020:
                categories['Moderate'] += 1
            else:
                categories['High'] += 1
    
    return categories
   