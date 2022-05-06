from django.db import models


# Create your models here.
class Image(models.Model):
    name = models.CharField(max_length=100)
    Main_Img = models.ImageField(upload_to="images/")
