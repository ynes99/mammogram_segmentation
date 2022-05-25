from django.db import models


# Create your models here.
class UploadedImage(models.Model):
    title = models.CharField(max_length=100, help_text='Enter an image title')
    image = models.ImageField(upload_to='images/')
    auto_increment_id = models.AutoField(primary_key=True)

    def __str__(self):
        return self.title


class segment(models.Model):
    any_field = models.CharField(max_length=20)
