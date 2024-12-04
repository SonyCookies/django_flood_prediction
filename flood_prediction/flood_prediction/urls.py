"""
URL configuration for flood_prediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from web_interface import views as web_views
from prediction.views import predict, predict_combined, predict_next_30_days

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', web_views.dashboard, name='dashboard'),
    path('api/predict', predict, name='predict'),
    path('predict', web_views.predict_flood, name='predict_flood'),
    path('api/predict_combined', predict_combined, name='predict_combined'),
    path('history', web_views.history, name='history'),
    path('api/history', web_views.get_history, name='get_history'),
    path('api/correlation', web_views.get_correlation_data, name='get_correlation_data'),
    path('correlation', web_views.correlation_relationship, name='correlation_relationship'),
    path('api/trend', web_views.get_trend_data, name='get_trend_data'),
    path('api/category', web_views.get_category_data, name='get_category_data'),
    path('api/earthquake', predict_next_30_days, name='predict_next_30_days'),
    path('earthquake', web_views.earthquake, name='earthquake'),
]
