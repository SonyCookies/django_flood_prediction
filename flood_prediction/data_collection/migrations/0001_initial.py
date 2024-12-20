# Generated by Django 5.1.3 on 2024-11-19 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RiverData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateTimeField()),
                ('river_discharge', models.FloatField()),
                ('river_discharge_mean', models.FloatField()),
                ('river_discharge_median', models.FloatField()),
                ('river_discharge_max', models.FloatField()),
                ('river_discharge_min', models.FloatField()),
                ('river_discharge_p25', models.FloatField()),
                ('river_discharge_p75', models.FloatField()),
            ],
        ),
    ]
