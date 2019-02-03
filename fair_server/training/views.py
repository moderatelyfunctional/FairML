from django.http import HttpResponse
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def play(request):
	empty_context = dict()
	return render(request, 'index.html', empty_context)

@csrf_exempt
def add_candidate(request):
	print('Request.POST is {}'.format(request.POST))
	print('Request.FILE is {}'.format(request))
	
	return HttpResponse('Hey everything good')

def train(request):
	data = request.POST.get('data')
	pass
