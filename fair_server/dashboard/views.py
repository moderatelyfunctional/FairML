import csv
import json

from django.http import HttpResponse

def fetch_n_resumes(request):
	n_resume = 0
	with open('dashboard/n_resumes.txt', 'r') as file:
		for line in file.readlines():
			n_resume = int(line)
	return HttpResponse(json.dumps({'n_resumes': n_resume}), content_type='application/json')

def fetch_csv(request):
	output_data = []
	index = 0
	with open('training/resume_data_5000.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			if index > 8:
				break
			output_data.append([row[1], row[2], row[3], row[4], row[5], row[6]])
			index += 1
	return HttpResponse(json.dumps(output_data), content_type='application/json')
