#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 7 2022
@author: Nassir Mohammad
"""

# %% setup
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

from rendering_functions import highlight_max, highlight_min
from utilities import show_results
from utilities import apply_classifiers, format_df
from utilities import highlight_max_min
from perception_nassir import Perception

# from pyod.models.auto_encoder import AutoEncoder
# import dataframe_image as dfi

# decide is plot images are to be saved
image_save_switch = False
image_save_path = (
    '/Users/nassirmohammad/Google Drive/docs/'
    'Anomaly_detection_using_principles_of_human_perception/'
    'figures/')

# path to research paper figures folder
latex_table_save_path = (
    '/Users/nassirmohammad/Google Drive/docs/'
    'Anomaly_detection_using_principles_of_human_perception/tables/')


# %% ex8data1
# Andrew Ng. Machine learning: Programming Exercise 8: Anomaly Detection and
# Recommender Systems. 2012.

data_path = r"../data/ex8data1.mat"
data = loadmat(data_path)
X = data['X']
X_val = data['Xval']
y_val = data['yval']

# show the headers
for key, val in data.items():
    print(key)

print("The shape of X is: {}".format(X.shape))
print("The shape of X_val is: {}".format(X_val.shape))
print("The shape of y is: {}".format(y_val.sum()))

# %% Standarise the validation data only (training data not used)
sc = StandardScaler()
sc.fit(X_val)
X_val = sc.transform(X_val)

# %% plot the data

# sns.set_theme()
fig, axs = plt.subplots(4, figsize=(12, 8))
sns.histplot(data=X[:, 0], ax=axs[0])
sns.histplot(data=X[:, 1], ax=axs[1])
sns.histplot(data=X_val[:, 0], ax=axs[2])
sns.histplot(data=X_val[:, 1], ax=axs[3])

# %% plot the training data
fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.scatterplot(x=X[:, 0], y=X[:, 1])
ax.set_title("Training data: no labels")

# %% plot the labelled validation data
X_val_anomalies = X_val[np.where(y_val == 1)[0]]
X_val_normal = X_val[np.where(y_val == 0)[0]]

fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.scatterplot(x=X_val_normal[:, 0], y=X_val_normal[:, 1], alpha=0.6)
ax = sns.scatterplot(x=X_val_anomalies[:, 0], y=X_val_anomalies[:, 1],
                     color='darkorange', alpha=1)
ax.set_title("Validation data (labeled)")


# =============================================================================
# Apply individual model to see output results
# =============================================================================

# %% MCD model
clf_mcd = MCD()
clf_mcd.fit(X_val)
validation_labels = clf_mcd.predict(X_val)
scores = clf_mcd.decision_function(X_val)

show_results(X_val, y_val, validation_labels,
             font_size="12",
             file_to_save=image_save_path + 'ex8data1_validation_mcd.png',
             figure_title='',  # "MCD results (validation data only)",
             decision_scores=scores,
             save_switch=image_save_switch)

# %% PCA model
clf_pca = PCA(n_selected_components=2)
clf_pca.fit(X_val)
validation_labels = clf_pca.predict(X_val)
scores = clf_pca.decision_function(X_val)

show_results(X_val, y_val, validation_labels,
             font_size="12",
             file_to_save=image_save_path + 'ex8data1_validation_pca.png',
             figure_title='',  # "PCA results (validation data only)",
             decision_scores=scores,
             save_switch=image_save_switch)

# %% Perception model
clf_perception = Perception()
clf_perception.fit(X_val)
validation_labels = clf_perception.predict(X_val)

show_results(X_val, y_val, validation_labels,
             font_size="12",
             file_to_save=image_save_path+'ex8data1_validation_perception.png',
             figure_title='',  # "Perception results (validation data only)",
             decision_scores=clf_perception.scores_,
             save_switch=image_save_switch)

# %% DBSCAN model
clf_dbscan = DBSCAN()
clustering = clf_dbscan.fit(X_val)
validation_labels = np.where(clustering.labels_ == -1, 1, 0)

show_results(X_val, y_val, validation_labels, font_size="12",
             file_to_save=image_save_path+'ex8data1_validation_dbscan.png',
             figure_title="DBSCAN results (validation data only)",
             decision_scores=None,
             save_switch=image_save_switch)

# %% Autoencoder with default parameters
# tensorflow issues with python 3.9 todo in future
# contamination = 0.1
# epochs = 30
# num_layers = 3
# # h1_temp = int(0.5 * X_val.shape[1])
# # h1 = min(h1_temp, 6)

# clf_AE = AutoEncoder(epochs=epochs, contamination=contamination,
#                      hidden_neurons=[2, 2, 2], verbose=0)

# clf_AE.hidden_neurons = [1]

# # prediction
# clf_AE.fit(X_val)
# validation_labels = clf_AE.predict(X_val)

# show_results(
# y_val, validation_labels, clf_AE.decision_scores_, font_size="12",
#              file_to_save='ex8data1_validation_ae.png',
#             figure_title="")

# %% apply all models
dataset_name = 'ex8data1'
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

df_temp = apply_classifiers(classifiers, dataset_name,
                            predict_data=X_val,
                            predict_labels=y_val,
                            train_data=X_val)

df = format_df(df_temp)

# %% format for highlighting max/min column values as image
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

# save the dataframe image file with highlights
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
    latex_table_save_path + 'ex8data1_results.txt',
    escape=False,
    index=False)

# %%
