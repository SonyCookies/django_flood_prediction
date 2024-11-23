from django.core.management.base import BaseCommand
from django_apscheduler.models import DjangoJob, DjangoJobExecution
from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore

class Command(BaseCommand):
    help = 'Removes all scheduled jobs and their executions'

    def handle(self, *args, **options):
        # Remove all job executions
        DjangoJobExecution.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Removed all job executions'))

        # Remove all jobs
        DjangoJob.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Removed all jobs'))

        # Clear the current scheduler
        scheduler = BackgroundScheduler()
        scheduler.add_jobstore(DjangoJobStore(), "default")
        scheduler.remove_all_jobs()
        
        self.stdout.write(self.style.SUCCESS('Successfully removed all scheduled jobs'))