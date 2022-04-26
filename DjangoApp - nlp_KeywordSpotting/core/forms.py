from django.forms import ModelForm
from .models import Audio

class AudioForm(ModelForm):
    class Meta:
        model = Audio
        fields = '__all__'
        exclude = ['classification']
