from django.db import models

class RiverData(models.Model):
    date = models.DateField()  # Use DateField instead of DateTimeField since the data is daily
    river_discharge = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "River Data"
        verbose_name_plural = "River Data"
        unique_together = ('date',)  # Ensure no duplicate entries for the same date

    def __str__(self):
        return f"River Data for {self.date}"

class WeatherData(models.Model):
    date = models.DateField()  # Align with the RiverData date for easier merging
    humidity = models.FloatField(null=True, blank=True)
    temperature = models.FloatField(null=True, blank=True)
    pressure = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "Weather Data"
        verbose_name_plural = "Weather Data"
        unique_together = ('date',)  # Ensure no duplicate entries for the same date

    def __str__(self):
        return f"Weather Data for {self.date}"

class EarthquakeData(models.Model):
    date = models.DateTimeField(db_index=True)
    magnitude = models.FloatField()
    depth = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    location = models.CharField(max_length=255)

    class Meta:
        unique_together = ('date', 'latitude', 'longitude')

    def __str__(self):
        return f"Earthquake at {self.location} on {self.date}"

