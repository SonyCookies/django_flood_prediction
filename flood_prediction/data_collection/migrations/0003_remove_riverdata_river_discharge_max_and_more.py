# Generated by Django 5.1.3 on 2024-11-21 14:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data_collection', '0002_alter_riverdata_options_alter_riverdata_date_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_max',
        ),
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_mean',
        ),
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_median',
        ),
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_min',
        ),
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_p25',
        ),
        migrations.RemoveField(
            model_name='riverdata',
            name='river_discharge_p75',
        ),
    ]
