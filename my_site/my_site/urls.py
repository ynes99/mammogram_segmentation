from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

app_name = 'my_app'

urlpatterns = [
    path('admin', admin.site.urls),
    path('', include("image.urls", namespace=app_name)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
