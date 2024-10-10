from django.urls import path
from .views import JobDatasetAnalysisAPIView

urlpatterns = [
    path('analyze_dataset/', JobDatasetAnalysisAPIView.as_view(), name='job_dataset_analysis'),
]
