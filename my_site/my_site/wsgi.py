"""
WSGI config for my_site project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_site.settings')
import sys
#Wrong!
#sys.path.append("/home/user/mysite/mysite")

#Correct
sys.path.append(r"C:\Users\Asus\PycharmProjects\pythonProject1\my_site")
application = get_wsgi_application()
