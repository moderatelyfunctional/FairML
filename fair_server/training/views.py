import PyPDF2

from django.http import HttpResponse
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def play(request):
	empty_context = dict()
	return render(request, 'index.html', empty_context)

@csrf_exempt
def add_candidate(request):
	for filename in request.FILES:
		print("FILENAME {}".format(request.FILES[filename]))
		name = request.FILES[filename].name
		data = request.FILES[filename].read()
		# print('data is {}'.format(data))
		# print('filename {} name {}'.format(filename, name))
	# print("Is there a candidate_pdf {}".format(candidate_pdf))
	
	candidate_pdf = request.FILES['candidate_pdf']
	print('Candidate pdf is {}'.format(candidate_pdf))
	
	return HttpResponse('Hey everything good')

def train(request):
	data = request.POST.get('data')
	pass
