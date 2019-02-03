import csv
import json

from django.http import HttpResponse

def fetch_csv(request):
	output_data = []
	index = 0
	with open('training/resume_data_5000.csv') as csv_file:
		csv_reader = csv.reader(csv_file, ',')
		for row in csv_reader:
			if index > 8:
				break
			output_data.append([row['Name'], row['GPA'], row['Gender'], row['Experience (yrs)'], row['School'], row['Accepted']])
			index += 1
	return HttpResponse(json.dumps(output_data), content_type='application/json')