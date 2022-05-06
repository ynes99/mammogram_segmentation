from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

app_name = 'my_app'
urlpatterns = [
    # path('image_upload/', image_view, name='image_upload'),
    path('success/', success, name='success'),
    path('', index, name='index'),
    path('symptoms/', symptoms, name='symptoms'),
    path('detection/', image_view, name='detection'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
