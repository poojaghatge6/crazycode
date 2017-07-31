from django.conf.urls import url
from . import views


app_name = 'pandp'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^form/$', views.Form, name='form'),
	url(r'^upload/$', views.Upload, name='upload'),
	url(r'^download/$', views.Download, name='download'),
	
	]