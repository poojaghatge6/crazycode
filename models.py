from django.db import models

# Create your models here.

class Vendor(models.Model):
     name=models.CharField(max_length=250)
     city=models.CharField(max_length=250)
     phonenumber = models.IntegerField()
     email_id = models.CharField(max_length = 50)

class Item(models.Model):
    description=models.CharField(max_length=1000)

class Purchase(models.Model):
    vendor=models.ForeignKey(Vendor, on_delete=models.CASCADE)
    item=models.ForeignKey(Item, on_delete=models.CASCADE)
