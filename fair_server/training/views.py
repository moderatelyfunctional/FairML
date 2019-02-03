from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def play(request):
	empty_context = dict()
	return render(request, 'index.html', empty_context)

def add_candidate(request):
	print("REQUEST IS {}".format(request.POST))
	
	return HttpResponse('Hey everything good')

def train(request):
	data = request.POST.get('data')
	pass
