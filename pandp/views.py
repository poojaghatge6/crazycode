from django.shortcuts import render
import pandas as pd

# Create your views here.
from django.http import HttpResponse
import csv

def index(request):
	return HttpResponse("<h1>This is music app homepage</h1>")

def Form(request):
	return render(request, "pandp/form.html", {})
	
def Upload(request):
	files = request.FILES.getlist('files')
	a = files[0]
	m = pd.read_csv(a)
	for f in files[1:]:
		b = pd.read_csv(f)
		b = b.dropna(axis=1)
		m = m.merge(b, on='title')
	m.to_csv('output.csv', index=False)
	print ("output.csv created")
	return render(request, "pandp/formDownload.html", {})

def Download(request):
	filename = settings.MEDIA_ROOT +'/'+ 'output.csv'
	#filename= r"C:\Users\A6B0SZZ\PycharmProjects\sample\media\output1.csv"
	download_name ="output.csv"		
	wrapper = FileWrapper(open(filename))
	response = HttpResponse(wrapper,content_type='text/csv')
	response['Content-Disposition'] = "attachment; filename=%s"%download_name
	return response
		
	
 


#class FileFieldView(FormView):
#	form_class = FileFieldForm
#	template_name = 'uploadSheet.html'  # Replace with your template.
#	success_url = 'pandp.views.index'  # Replace with your URL or reverse().
#
#	def post(self, request, *args, **kwargs):
#		form_class = self.get_form_class()
#		form = self.get_form(form_class)
#		filename = settings.MEDIA_ROOT +'/'+ 'output.csv'
		#filename= r"C:\Users\A6B0SZZ\PycharmProjects\sample\media\output1.csv"
#		download_name ="output.csv"		
#		wrapper = FileWrapper(open(filename))
#		response = HttpResponse(wrapper,content_type='text/csv')
#		response['Content-Disposition'] = "attachment; filename=%s"%download_name
#		return response

			
	