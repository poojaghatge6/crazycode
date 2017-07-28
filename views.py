from django.http import HttpResponse

def index(request):
    return HttpResponse("<h1>This is music app homepage</h1>")
