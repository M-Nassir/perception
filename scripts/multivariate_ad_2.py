#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:50:23 2022
@author: Nassir, Mohammad
"""
# %% setup
from perception import Perception
from utilities import apply_classifiers, format_df
from utilities import highlight_max_min
from rendering_functions import highlight_max, highlight_min

import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
import seaborn as sns

from pyod.models.mcd import MCD
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import dataframe_image as dfi

image_save_switch = True
image_save_path = (
    '/Users/nassirmohammad/Google Drive/docs/'
    'Anomaly_detection_using_principles_of_human_perception/figures/')

latex_table_save_path = (
    '/Users/nassirmohammad/Google Drive/docs/'
    'Anomaly_detection_using_principles_of_human_perception/tables/')

# %% ex8data2
# Andrew Ng. Machine learning: Programming Exercise 8: Anomaly Detection and
# Recommender Systems. 2012.

data_path = r"../data/ex8data2.mat"
data = loadmat(data_path)
X2 = data['X']
X2_val = data['Xval']
y2_val = data['yval']

# show the headers
for key, val in data.items():
    print(key)

print("The shape of X2 is: {}".format(X2.shape))
print("The shape of X2_val is: {}".format(X2_val.shape))
print("The shape of y2 is: {}".format(y2_val.sum()))

# %% standardise the data
sc = StandardScaler()
sc.fit(X2)
X2 = sc.transform(X2)
X2_val = sc.transform(X2_val)

# %% plot the data
df = pd.DataFrame(X2)
sns.pairplot(df)

# view the validation data with labels
df2 = pd.DataFrame(np.append(X2_val, y2_val, 1))
sns.pairplot(df2, hue=11)

# %% Implement a model: MCD
clf_mcd = MCD()
clf_mcd.fit(X2)
validation_labels = clf_mcd.predict(X2_val)
scores = clf_mcd.decision_function(X2_val)

print(classification_report(
    y2_val, validation_labels,
    target_names=['normal', 'abnormal'], output_dict=False))

auc_score = roc_auc_score(y2_val, scores)
print('auc score: {}'.format(auc_score))

# %% implement a model: Perception
clf_perception = Perception()
clf_perception.fit(X2)
clf_perception.predict(X2_val)

validation_labels = clf_perception.labels_

print(classification_report(
    y2_val, validation_labels,
    target_names=['normal', 'abnormal'], output_dict=False))

auc_score = roc_auc_score(y2_val, clf_perception.scores_)
print('auc score: {}'.format(auc_score))

# %% implement a model DBSCAN
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    clf_dbscan = DBSCAN()

    # likely not enough data to be used this way
    clustering = clf_dbscan.fit(X2_val)

    validation_labels = np.where(clustering.labels_ == -1, 1, 0)

    print(classification_report(
        y2_val, validation_labels,
        target_names=['normal', 'abnormal'], output_dict=False))

    auc_score = roc_auc_score(y2_val, validation_labels)
    print('auc score: {}'.format(auc_score))

# %% implement a model: Autoencoder
# need to sort out tensorflow compatibility with python 3.9

# data = loadmat(path)
# X2 = data['X']
# X2_val = data['Xval']
# y2_val = data['yval']

# contamination = 0.1
# epochs = 30

# num_layers = 3
# h1 = int(0.5 * X_val.shape[1])
# h2 = max(int(0.5 * h1), 1)

# clf_AE = AutoEncoder(epochs=epochs,
# contamination=contamination, hidden_neurons = [h1, h2, h1], verbose=0)

# # prediction
# clf_AE.fit(X2)
# validation_labels = clf_AE.predict(X2_val)

# print(classification_report(y2_val,
# validation_labels, target_names=['normal', 'abnormal'],output_dict=False))

# auc_score = roc_auc_score(y2_val, validation_labels)
# print('auc score: {}'.format(auc_score))

# %% Apply all models
dataset_name = 'ex8data2'
classifiers = [
    'HBOS',  # to be ignored, first run in loop slower
    'HBOS',
    'IForest',
    'KNN',
    'LOF',
    'MCD',
    'OCSVM',
    'Perception',
    # 'DBSCAN',
]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    df_temp = apply_classifiers(classifiers, dataset_name,
                                predict_data=X2_val,
                                predict_labels=y2_val,
                                train_data=X2)

df = format_df(df_temp)

# %% format for highlighting max/min column values
cols = ['Precision', 'Recall', 'F1', 'AUC']
formatdict = {}
for col in cols:
    formatdict[col] = "{:.2f}"
formatdict.pop('Classifier', None)
formatdict['Runtime'] = "{:.5f}"

metrics_df_styled = df.style.hide_index().\
    apply(highlight_max,
          subset=['Precision', 'Recall', 'F1',
                  'AUC']).\
    apply(highlight_min, subset=['Runtime']).format(formatdict)

# title_img = 'Anomaly detection results on dataset ex8data1'
# path = image_save_path + 'ex8data1_validation_table.png'

# dfi.export(metrics_df_styled, path)

# %% latex format print and save to file
col_show_max = {"Precision": True,
                "Recall": True,
                "F1": True,
                "AUC": True,
                "Runtime": False}

highlighted_df = highlight_max_min(df, col_show_max)

print(highlighted_df.to_latex(escape=False, index=False))

# write the latex formatted table to paper location
highlighted_df.to_latex(
    latex_table_save_path + 'ex8data2_results.txt',
    escape=False,
    index=False)
