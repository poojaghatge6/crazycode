#!/usr/bin/python

# Import modules for CGI handling 
import cgi, cgitb 

# Create instance of FieldStorage 


import pandas as pd

form = cgi.FieldStorage()

files = request.FILES.getlist('myfiles')
		a = files[0]
		if form.is_valid():
			for f in files:
				b = pd.read_csv(f)
				b = b.dropna(axis=1)
				a= a.merge(b, on='title')
			a.to_csv('output.csv', index=False)