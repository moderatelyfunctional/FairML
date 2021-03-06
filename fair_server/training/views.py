import os
import json

import PyPDF2

from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt

import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

np.random.seed(0)

# Create your views here.
def play(request):
    empty_context = dict()
    return render(request, 'index.html', empty_context)

name_to_gender = {
    'Arisa Pono': 'F',
    'Olivia Dzumade': 'F',
    'Joseph Adams': 'M'
}
@csrf_exempt
def add_candidate(request):
    # curr_file = request.FILES['candidate_pdf']
    n_resume = 0
    with open('dashboard/n_resumes.txt', 'r') as file:
        for line in file.readlines():
            n_resume = int(line)

    with open('dashboard/n_resumes.txt', 'w') as file:
        file.write(n_resume + 1)

    curr_file = request.FILES['image']

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
        'Name': name,
        'Gender': gender,
        'School': school,
        'GPA': gpa,
        'Experience (yrs)': experience
    }
    prediction = predict(data)
    output_data = {
        'data': data,
        'prediction': prediction
    }
    del data['Experience (yrs)']
    data['Ex'] = experience
    return HttpResponse(json.dumps(output_data), content_type='application/json')

def predict(data):
    with open("./training/model_orig.pkl", 'rb') as f:
        model_orig = pickle.load(f)
    with open("./training/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open("./training/metrics_orig.pkl", 'rb') as f:
        metrics_orig = pickle.load(f)
    with open("./training/model_transf.pkl", 'rb') as f:
        model_transf = pickle.load(f)
    with open("./training/metrics_transf.pkl", 'rb') as f:
        metrics_transf = pickle.load(f)

    X = np.array(data_to_vector(data))
    X_scaled = scaler.transform(X.reshape(1,-1))
    y_orig = model_orig.predict(X_scaled)
    no_fair_rec = 'Accept' if (y_orig == 1) else 'Reject'
    y_transf = model_transf.predict(X_scaled)
    fair_rec = 'Accept' if (y_transf == 1) else 'Reject'

    output_data = {
        'no_fair_acc': metrics_orig['Accuracy (Overall)'],
        'no_fair_bias': metrics_orig['Average odds difference'],
        'no_fair_rec': no_fair_rec,
        'fair_acc': metrics_transf['Accuracy (Overall)'],
        'fair_bias': metrics_transf['Average odds difference'],
        'fair_rec': fair_rec
    }
    return output_data

def data_to_vector(data):
    schools_list = ['Penn', 'Barnard', 'Wellesley', 'MIT', 'USC', 'Cornell']

    vector = [0] * 3
    vector[0] = data['GPA']
    vector[1] = 1 if data['Gender'] == 'M' else 0
    vector[2] = data['Experience (yrs)']
    
    idx = schools_list.index(data['School'])
    se = [0] * 6
    se[idx] = 1

    vector = vector + se
    return vector

# Giant function based on model Python notebook
def train(request):
    df = pd.read_csv('./training/resume_data_5000.csv')
    df = df.drop(df.columns[0], axis=1)
    dataset_orig = StandardDataset(df,
                                    label_name='Accepted',
                                    favorable_classes=[1],
                                    protected_attribute_names=['Gender'],
                                    privileged_classes=[[1]], categorical_features=['School'],
                                    features_to_drop=['Name'])
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

    privileged_groups = [{'Gender': 1}]
    unprivileged_groups = [{'Gender': 0}]

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    orig_mean_difference = metric_orig_train.mean_difference()

    with open('./training/orig_mean_difference.pkl', 'wb') as f:
        pickle.dump(orig_mean_difference, f)

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)
    transf_mean_difference = metric_transf_train.mean_difference()

    with open('./training/transf_mean_difference.pkl', 'wb') as f:
        pickle.dump(transf_mean_difference, f)

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    w_train = dataset_orig_train.instance_weights.ravel()

    with open('./training/scaler.pkl', 'wb') as f:
        pickle.dump(scale_orig, f)

    lmod_orig = LogisticRegression(solver='lbfgs')
    lmod_orig.fit(X_train, y_train,
            sample_weight=dataset_orig_train.instance_weights)
    y_train_pred = lmod_orig.predict(X_train)

    pos_ind = np.where(lmod_orig.classes_ == dataset_orig_train.favorable_label)[0][0]

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod_orig.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod_orig.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):

        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                dataset_orig_valid_pred,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                        +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    for thresh in tqdm(class_thresh_arr):
        fav_inds = dataset_orig_test_pred.scores > thresh
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

        metric_test_bef = compute_metrics(dataset_orig_test, dataset_orig_test_pred,
                                        unprivileged_groups, privileged_groups,
                                        disp = False)

        if thresh == best_class_thresh:
            with open('./training/metrics_orig.pkl', 'wb') as f:
                pickle.dump(metric_test_bef, f, protocol=pickle.HIGHEST_PROTOCOL)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_train.features)
    y_train = dataset_transf_train.labels.ravel()

    lmod_transf = LogisticRegression(solver='lbfgs')
    lmod_transf.fit(X_train, y_train,
            sample_weight=dataset_transf_train.instance_weights)
    y_train_pred = lmod_transf.predict(X_train)

    dataset_transf_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_transf.fit_transform(dataset_transf_test_pred.features)
    y_test = dataset_transf_test_pred.labels
    dataset_transf_test_pred.scores = lmod_transf.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    bal_acc_arr_transf = []
    disp_imp_arr_transf = []
    avg_odds_diff_arr_transf = []

    for thresh in tqdm(class_thresh_arr):
        fav_inds = dataset_transf_test_pred.scores > thresh
        dataset_transf_test_pred.labels[fav_inds] = dataset_transf_test_pred.favorable_label
        dataset_transf_test_pred.labels[~fav_inds] = dataset_transf_test_pred.unfavorable_label

        metric_test_aft = compute_metrics(dataset_orig_test, dataset_transf_test_pred,
                                        unprivileged_groups, privileged_groups,
                                        disp = False)

        if thresh == best_class_thresh:
            with open('./training/metrics_transf.pkl', 'wb') as f:
                pickle.dump(metric_test_aft, f, protocol=pickle.HIGHEST_PROTOCOL)

        bal_acc_arr_transf.append(metric_test_aft["Balanced accuracy"])
        avg_odds_diff_arr_transf.append(metric_test_aft["Average odds difference"])
        disp_imp_arr_transf.append(metric_test_aft["Disparate impact"])

    with open('./training/model_orig.pkl', 'wb') as f:
        pickle.dump(lmod_orig, f)
    with open('./training/model_transf.pkl', 'wb') as f:
        pickle.dump(lmod_transf, f)

    return HttpResponse('Model trained')


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Accuracy (Overall)"] = classified_metric_pred.accuracy()
    metrics["Accuracy (Privileged)"] = classified_metric_pred.accuracy(True)
    metrics["Accuracy (Unprivileged)"] = classified_metric_pred.accuracy(False)
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics
