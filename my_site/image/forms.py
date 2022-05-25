from django.forms import *
from .models import *


class UploadImageForm(ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image', 'title', 'auto_increment_id']


class MyModelForm(ModelForm):
    class Meta:
        model = segment
        fields = '__all__'
        widgets = {'any_field': HiddenInput(), }
