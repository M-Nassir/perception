#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:39:20 2022

@author: Nassir Mohammad
"""

# %% imports and setup
import os
import warnings
import numpy as np
import pandas as pd
from utilities import get_file_data, highlight_max_min
from utilities import apply_classifiers
from sklearn.preprocessing import StandardScaler

# from pyod.models.auto_encoder import AutoEncoder
# import dataframe_image as dfi

image_save_path = ''
image_save_switch = False

latex_table_save_path = (
    r"/Users/nassirmohammad/Google Drive/docs/"
    r"Anomaly_detection_using_principles_of_human_perception/"
    r"tables/")

# %% get properties of each data file adn write df to latex
base_path = "../data/ODDS_multivariate/"
data_properties_df = None

# loop over datasets in directory
for file_name in os.listdir(base_path):

    dataset_name, X_original, y = get_file_data(base_path, file_name)

    # if file name is not supported, just continue
    if dataset_name is None:
        continue

    # write dataset summary to dataframe
    data_properties_temp = pd.DataFrame({
        'Name': [dataset_name],
        '\\# examples': [X_original.shape[0]],
        '\\# features': [X_original.shape[1]],
        '\\% anomalies': [float(np.round(y.sum()/X_original.shape[0]*100, 2))],
    })

    data_properties_df = pd.concat(
        [data_properties_df, data_properties_temp]).reset_index(drop=True)

# write the latex formatted table to paper location
data_properties_df.to_latex(
    latex_table_save_path + 'dataset_properties.txt',
    escape=False,
    index=False)

# %% save dataframe as image
# img_title = "Dataset properties"
# path_save = image_save_path + "dataset_properties.png"

# # order the dataset rows by name
# meta_data = meta_data.sort_values(by=['Name']).reset_index(drop=True)

# meta_data_styled = meta_data.style.format(
#     {'% anomalies': "{:.2f}"}).hide_index()

# meta_data_styled = meta_data.style.hide_index()

# dfi.export(meta_data,path_save)

# %% apply classifiers to each dataset

# ########################
# file names used in paper
# ########################

# file_name = "wbc.mat"
# file_name = "cardio.mat"
# file_name = "thyroid.mat"
# file_name = "musk.mat"
# file_name = "shuttle.mat"
# file_name = "satimage-2.mat"
# file_name = "http.matv7"
# file_name = "smtp.matv7"
# file_name = "creditcard.csv"

# #########################
# data sets to explore
# #########################

# good dataset with differing performance, but chose Rayana
# file_name = "breast-cancer-unsupervised-ad.csv"

# seems all categorial data
# file_name = "lympho.mat"

# anomalies do not appear to be seperable as all classifier give poor results
# file_name = "glass.mat"

# appears to be a local outlier problem with anomalies in between normal points
# file_name = "vowels.mat"
# file_name = "cover.mat"
# gives different results
# file_name = "annthyroid-unsupervised-ad.csv"

# this is image data
# file_name = "mammography.mat" # this is bi-modal

# % of anomalies is more than 10%
# file_name = "arrhythmia.mat"

# file_name = "satellite.mat"
# file_name = "satellite-unsupervised-ad.csv"
# file_name = "smtp.mat"
# file_name = "kdd99-unsupervised-ad.csv"


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

metrics_df = None

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    # loop over datasets in directory
    for file_name in os.listdir(base_path):

        dataset_name, X_original, y = get_file_data(base_path, file_name)

        if dataset_name is None:
            continue

        # scaling (very important to get right)
        # scale to zero mean and unit standard deviation along each feature
        sc = StandardScaler(with_mean=False)
        sc.fit(X_original)
        X = sc.transform(X_original)

        # Apply each classifier to dataset
        print('current file in progress ...: {}'.format(dataset_name))

        metrics_temp = apply_classifiers(classifiers, dataset_name,
                                         predict_data=X,
                                         predict_labels=y,
                                         train_data=X)

        metrics_df =\
            pd.concat([metrics_df, metrics_temp])

    metrics_df.reset_index(drop=True)

# %% test code
# from sklearn.cluster import DBSCAN
# import time
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')

#     dataset_name, X_original, y = get_file_data(base_path,'credit-card.csv')
#     dataset_name, X_original, y = get_file_data(base_path, 'shuttle.mat')

#     sc = StandardScaler(with_mean=False)
#     sc.fit(X_original)
#     X = sc.transform(X_original)

#     s = time.perf_counter()
#     clf = DBSCAN()
#     clf.fit(X)
#     # clf.predict(X)
#     # clf.decision_function(X)
#     # e = time.perf_counter()
#     # print(e-s)


# %% get metric results and save to latex


def get_metrics_save_to_latex(df_input, pivot_col, save_path,
                              round_val, col_max=True):

    df_t = df_input[['Dataset', 'Classifier', pivot_col]]
    df_t = pd.pivot_table(df_t, values=pivot_col, index=[
        'Classifier'], columns='Dataset').reset_index()
    df_t.columns.name = None
    df_t = df_t.round(round_val)

    # save to latex and highlight max
    column_subset = df_t.columns.values.tolist()
    column_subset.remove('Classifier')

    if col_max:
        col_show_max = {i: True for i in column_subset}
    else:
        col_show_max = {i: False for i in column_subset}

    highlighted_df = highlight_max_min(df_t, col_show_max)

    print(highlighted_df.to_latex(escape=False, index=False))

    # write the latex formatted table to paper location
    highlighted_df.to_latex(
        save_path,
        escape=False,
        index=False)


df_precision = get_metrics_save_to_latex(
    metrics_df, pivot_col='Precision',
    save_path=latex_table_save_path + 'precision_results.txt',
    round_val=3, col_max=True)


df_precision = get_metrics_save_to_latex(
    metrics_df, pivot_col='Recall',
    save_path=latex_table_save_path + 'recall_results.txt',
    round_val=2, col_max=True)

df_precision = get_metrics_save_to_latex(
    metrics_df, pivot_col='F1',
    save_path=latex_table_save_path + 'f1_results.txt',
    round_val=3, col_max=True)

df_precision = get_metrics_save_to_latex(
    metrics_df, pivot_col='AUC',
    save_path=latex_table_save_path + 'auc_results.txt',
    round_val=2, col_max=True)

df_precision = get_metrics_save_to_latex(
    metrics_df, pivot_col='Runtime',
    save_path=latex_table_save_path + 'runtime_results.txt',
    round_val=3, col_max=False)

# %%