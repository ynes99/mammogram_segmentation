from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import ImageForm


# Create your views here.
def image_view(request):
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = ImageForm()
    return render(request, "my_app/treatment.html", {"form": form})


def index(request):
    return render(request, "my_app/index.html")


def symptoms(request):
    return render(request, "my_app/symptoms.html")


def detection(request):
    return render(request, "my_app/treatment.html")


def success(request):
    return HttpResponse('successfully uploaded')
