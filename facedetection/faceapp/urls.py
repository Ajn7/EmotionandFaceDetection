from django.urls import path

from faceapp.views import faceapp_view

urlpatterns = [ 
               
    path('recognition/',faceapp_view,name='face_recognition'),  
]