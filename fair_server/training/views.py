import os
import base64

import PyPDF2

from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt

# Create your views here.
def play(request):
	empty_context = dict()
	return render(request, 'index.html', empty_context)

name_to_gender = {
	'Arisa Pono': 'F',
	'Efraim Helman': 'M',
	'Joseph Adams': 'M'
}
@csrf_exempt
def add_candidate(request):
	curr_file = request.FILES['candidate_pdf']

	output_pdf = '{}'.format(curr_file.name)
	with default_storage.open(output_pdf, 'wb+') as dest:
		dest.write(curr_file.read())

	pdf_file = open(output_pdf, 'rb')
	read_pdf = PyPDF2.PdfFileReader(pdf_file)
	page = read_pdf.getPage(0)
	page_content = page.extractText()

	text_content = page_content.split('\n')
	os.remove(output_pdf)

	name = ' '.join(text_content[0].split(' ')[:2])
	gender = name_to_gender[name]
	school = text_content[5][1:].strip()

	gpa = float(text_content[8].split(' ')[0][1:])

	award_index = 0
	for (index, text) in enumerate(text_content):
		if 'Awards' in text:
			award_index = index
			break

	experience = min(6, (award_index - 8) // 3)

	data = {
		'name': name,
		'gender': gender,
		'school': school,
		'gpa': gpa,
		'experience': experience
	}
	print("data is {}".format(data))
	return HttpResponse('Hey everything good')

def train(request):
	data = request.POST.get('data')
	pass
