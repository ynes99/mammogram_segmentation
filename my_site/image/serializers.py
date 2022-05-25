from rest_framework import serializers
from .models import UploadedImage, segment


class SegmentSerializers(serializers.ModelSerializer):
    class meta:
        model = segment
        fields = '__all__'
