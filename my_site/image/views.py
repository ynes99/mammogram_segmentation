from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import ImageForm
from .models import Image


# Create your views here.
def image_view(request):
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, "treatment.html", {"form": form})


def index(request):
    return render(request, "index.html")


def success(request):
    return HttpResponse('successfully uploaded')
