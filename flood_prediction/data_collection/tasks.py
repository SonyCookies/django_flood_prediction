import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from django.conf import settings
from django_apscheduler.jobstores import DjangoJobStore
from django.utils import timezone
from datetime import datetime, timedelta
import requests
import pytz
from django.db import transaction
from .models import RiverData, WeatherData, EarthquakeData
from model_trainer.tasks import train_models

logger = logging.getLogger(__name__)

def fetch_weather_data():
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    api_key = "JM4G6685UZS9QUD9XR3MPSN8F" 
    location = "13.27443,121.259722"
    start_date = '2024-11-24' #change this
    end_date = timezone.now().strftime('%Y-%m-%d')

    params = {
        "unitGroup": "metric",  
        "include": "days",  
        "key": api_key, 
        "contentType": "json"  
    }

    try:
        # API request
        response = requests.get(f"{url}/{location}/{start_date}/{end_date}", params=params)
        response.raise_for_status()
        data = response.json()

        # Process daily data
        daily_data = data['days']
        weather_data_objects = []
        for day in daily_data:
            avg_temp = (day['tempmax'] + day['tempmin']) / 2  # Calculate average temperature
            humidity = day.get('humidity', None)  # Fetch humidity
            pressure = day.get('pressure', None)  # Fetch pressure

            # Create WeatherData instance
            weather_data_objects.append(WeatherData(
                date=datetime.strptime(day['datetime'], '%Y-%m-%d').date(),
                temperature=avg_temp,
                humidity=humidity,
                pressure=pressure,
            ))

        # Bulk save data to database
        with transaction.atomic():
            WeatherData.objects.bulk_create(weather_data_objects, ignore_conflicts=True)

        logger.info(f"Successfully fetched and saved weather data from {start_date} to {end_date}")
        return f"Successfully fetched and saved weather data from {start_date} to {end_date}"

    except requests.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return f"Error fetching weather data: {e}"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"

def fetch_river_data():
    url = "https://flood-api.open-meteo.com/v1/flood"
    start_date = '2024-11-24' #change this
    end_date = timezone.now().strftime('%Y-%m-%d')
    
    params = {
        "latitude": 13.27443,
        "longitude": 121.259722,
        "daily": ["river_discharge"],
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        daily_data = data['daily']
        time_data = daily_data['time']

        river_data_objects = [
            RiverData(
                date=datetime.strptime(time, '%Y-%m-%d').date(),
                river_discharge=daily_data['river_discharge'][i],
            )
            for i, time in enumerate(time_data)
        ]

        with transaction.atomic():
            RiverData.objects.bulk_create(river_data_objects, ignore_conflicts=True)

        logger.info(f"Successfully fetched river data from {start_date} to {end_date}")
    except Exception as e:
        logger.error(f"Error fetching river data: {str(e)}")

def fetch_earthquake_data():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    end_date = timezone.now()
    start_date = end_date - timedelta(days=1)
    
    params = {
        "format": "geojson",
        "starttime": start_date.strftime('%Y-%m-%d'),
        "endtime": end_date.strftime('%Y-%m-%d'),
        "minlatitude": 4.5,
        "maxlatitude": 21.0,
        "minlongitude": 116.0,
        "maxlongitude": 127.0,
        "minmagnitude": 4.0
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        earthquake_data_objects = []
        for feature in data['features']:
            properties = feature['properties']
            coordinates = feature['geometry']['coordinates']
            
            if "Philippines" in properties.get('place', ''):
                earthquake_date = datetime.fromtimestamp(properties['time'] / 1000.0, tz=pytz.UTC)
                
                earthquake_data_objects.append(EarthquakeData(
                    date=earthquake_date,
                    magnitude=properties['mag'],
                    depth=coordinates[2],
                    latitude=coordinates[1],
                    longitude=coordinates[0],
                    location=properties['place']
                ))

        with transaction.atomic():
            EarthquakeData.objects.bulk_create(earthquake_data_objects, ignore_conflicts=True)

        logger.info(f"Successfully fetched and saved earthquake data for the Philippines from {start_date.date()} to {end_date.date()}")
        return f"Successfully fetched and saved earthquake data for the Philippines from {start_date.date()} to {end_date.date()}"

    except requests.RequestException as e:
        logger.error(f"Error fetching earthquake data: {e}")
        return f"Error fetching earthquake data: {e}"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error: {e}" 
    
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_jobstore(DjangoJobStore(), "default")

    scheduler.add_job(
        fetch_all_data,
        trigger=CronTrigger(hour="6", minute="0"),
        id="fetch_all_data",
        max_instances=1,
        replace_existing=True,
    )
    logger.info("Added job 'fetch_all_data'.")

    scheduler.add_job(
        train_models,
        trigger=CronTrigger(hour="7", minute="0"),
        id="train_models",
        max_instances=1,
        replace_existing=True,
    )
    logger.info("Added job 'train_models'.")

    try:
        logger.info("Starting scheduler...")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
        scheduler.shutdown()
        logger.info("Scheduler shut down successfully!")
def test_job():
    logger.info(f"Test job run at {timezone.now()}")

def fetch_all_data():
    logger.info("Fetching all data...")
    fetch_river_data()
    fetch_weather_data()
    logger.info("Finished fetching all data.")