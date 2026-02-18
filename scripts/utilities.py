'''
@author: Nassir Mohammad
'''

from scipy.io import loadmat
import h5py
import numpy as np
from scipy import stats
from perception_nassir import Perception
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import time
import os

# the below are used via eval()
from pyod.models.hbos import HBOS  # NOQA
from pyod.models.iforest import IForest  # NOQA
from pyod.models.knn import KNN  # NOQA
from pyod.models.lof import LOF  # NOQA
from pyod.models.mcd import MCD  # NOQA
from pyod.models.ocsvm import OCSVM  # NOQA
from pyod.models.pca import PCA  # NOQA
from sklearn.cluster import DBSCAN  # NOQA
import warnings  # NOQA

# used for numenta data


def show_anomalies(df, results, algorithm_label, alg_name,
                   column_name='value', time_column='timestamp'):

    # assign label for that row based on 'new_values'
    df[algorithm_label] = (df[column_name].isin(
        results.loc[alg_name])).astype(int)

    a = df.loc[df[algorithm_label] == 1, [time_column, column_name]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=time_column, y=column_name, data=df, ci=None)

    # rotate the labels
    ax.tick_params(axis='x', rotation=45)

    # set ticks every week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())

    # set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    ax.scatter(a[time_column], a[column_name], color='red')
    plt.show()

    # visualisation of anomaly with temperature repartition (viz 2)
    a = df.loc[df[algorithm_label] == 0, column_name]
    b = df.loc[df[algorithm_label] == 1, column_name]

    fig, axs = plt.subplots()
    axs.hist([a, b], bins=32, stacked=True, color=[
             'blue', 'red'], label=['normal', 'anomaly'],
             alpha=0.5)

    plt.legend()
    plt.show()

    # drop the temp label column
    df.drop(algorithm_label, axis=1, inplace=True)


def show_results(data, y_val, validation_labels,
                 font_size, figure_title,
                 file_to_save,
                 decision_scores=None,
                 save_switch=False):

    X_val_anomalies = data[np.where(validation_labels == 1)[0]]
    X_val_normal = data[np.where(validation_labels == 0)[0]]

    fp = data[np.where((validation_labels == 1) & (y_val.T == 0))[1]]
    fn = data[np.where((validation_labels == 0) & (y_val.T == 1))[1]]

    for x in fp:
        X_val_anomalies = np.delete(
            X_val_anomalies, np.where(X_val_anomalies == x), axis=0)

    tp = X_val_anomalies

    for x in fn:
        X_val_normal = np.delete(
            X_val_normal, np.where(X_val_normal == x), axis=0)

    tn = X_val_normal

    plt.rcParams["font.size"] = font_size

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.scatterplot(x=tp[:, 0], y=tp[:, 1], color='darkorange',
                         label='anomaly (true +ve)')
    ax = sns.scatterplot(x=tn[:, 0], y=tn[:, 1],
                         alpha=0.4, label='normal (true -ve)')
    plt.legend()

    ax = sns.scatterplot(x=fp[:, 0], y=fp[:, 1], color='purple',
                         label='normal (false +ve)', marker='x')
    ax = sns.scatterplot(x=fn[:, 0], y=fn[:, 1], color='black',
                         label='anomaly (false -ve)', marker=',')

    ax.set_title(figure_title, fontsize=18)
    plt.legend()

    if save_switch is True:
        plt.savefig(file_to_save)

    print(classification_report(y_val, validation_labels,
          target_names=['normal', 'abnormal'], output_dict=False))

    if decision_scores is None:
        # produce AUC score with only binary labels
        auc_score = roc_auc_score(y_val, validation_labels)
    else:
        # produce AUC score using classifier prediction scores
        auc_score = roc_auc_score(y_val, decision_scores)

    print('auc score: {}'.format(auc_score))


def plot_data(data, save_path=None, save_switch=False, balls_height=0.1,
              x_label='', y_label='', title=''):

    # seaborn histogram and scatter plot
    f, axes = plt.subplots(figsize=(18, 10))
    sns.histplot(data, kde=False, bins=10, color='grey',
                 alpha=0.2,  edgecolor='white'
                 )
    sns.scatterplot(x=data, y=np.zeros_like(data)+balls_height,
                    color='black',
                    marker='o',
                    edgecolor='black',
                    alpha=1, s=100)

    # Add labels
    plt.title(title)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.tick_params(labelsize=20)

    if save_switch is True:
        plt.savefig(save_path)

    plt.show()

# %% apply anomaly detection algorithms to 1D data


def apply_methods(data):
    methods = [perception_method, z_score_2_5_method,
               z_score_3_method, mad_method, iqr_method]

    results = run_methods(methods, data)
    results_df = pd.DataFrame.from_dict(results, orient='index')
    return results_df


def z_score_3_method(data):
    z = np.abs(stats.zscore(data))
    threshold = 3
    outlier_indices = np.where(z > threshold)
    return np.sort(data[outlier_indices])


def z_score_2_5_method(data):
    z = np.abs(stats.zscore(data))
    threshold = 2.5
    outlier_indices = np.where(z > threshold)
    return np.sort(data[outlier_indices])


def mad_method(data):
    threshold = 3.5
    median_ys = np.median(data)
    MAD_ys = np.median(np.abs(data - median_ys))

    if MAD_ys == 0:
        MAE_ys = (1 / len(data)) * np.sum(np.abs(data - median_ys))
        modified_z_scores = 0.7979 * ((data - median_ys) / MAE_ys)
    else:
        modified_z_scores = 0.6745 * ((data - median_ys) / MAD_ys)

    mad_outlier_indices = np.where(np.abs(modified_z_scores) > threshold)

    return np.sort(data[mad_outlier_indices])


def iqr_method(data):
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    boxplot_outlier_indices = \
        np.where((data > upper_bound) | (data < lower_bound))

    return np.sort(data[boxplot_outlier_indices])


def perception_method(data):
    clf = Perception()
    clf.fit(data)
    clf.predict(data)

    return np.sort(clf.anomalies_)


def run_methods(methods, data):
    """Run all the methods on the input data

    Parameters:
    methods (list of functions): The methods to run on the supplied data
    data (numpy 1-D array): The input data as a 1-D numpy array of numbers
    decimal_accuracy (int): The number of decimal places to consider for the
    input data for 'our' method only
    """

    # round the data to the required precision
    # data = np.round(data, decimals=raw_decimal_accuracy)

    # declare empty array to hold the results to be returned
    results = {}

    # loop over the methods and append the list of anomalies
    for method in methods:

        if method.__name__ == 'perception_method':
            result = method(data)
        else:
            result = method(data)

        # if the array is empty, assign string
        if result.size == 0:
            # 'No outliers detected'
            result = np.array([])

        # results.append(result)
        results.update({method.__name__: result})

    return results

# %% apply anomaly detection algorithms to multidimensional data


def call_init_classifier(function_name):

    start = time.perf_counter()
    result = eval(function_name + "()")
    end = time.perf_counter()
    init_time = end-start
    return result, init_time


def call_fit(clf, name, data):

    if name == 'DBSCAN':

        start = time.perf_counter()
        clustering = clf.fit(data)
        labels_temp = np.where(clustering.labels_ == -1, 1, 0)
        end = time.perf_counter()
        train_time = end-start

        return labels_temp, train_time

    else:
        s1 = time.perf_counter()
        clf = clf.fit(data)
        e1 = time.perf_counter()
        train_time = e1-s1

        return clf, train_time


def call_predict(clf, clf_name, data):

    start = time.perf_counter()
    labels = clf.predict(data)

    if clf_name == 'Perception':
        scores = clf.scores_
    else:
        scores = clf.decision_function(data)

    end = time.perf_counter()
    predict_time = end-start

    return labels, scores, predict_time


def apply_classifiers(classifiers, dataset_name,
                      predict_data, predict_labels,
                      train_data=None):

    metrics_df = None
    labels = np.empty(len(predict_data))
    scores = np.array(len(predict_data))

    # run the classifiers over the data and produce results
    for clf_name in classifiers:

        print('current classifier in progress: {}'.format(clf_name))

        if ((dataset_name == "http") and (clf_name == "OCSVM")):
            continue  # takes too long to run

        elif (dataset_name == "creditcard") and\
                (clf_name == "LOF" or
                 clf_name == "KNN" or
                 clf_name == "OCSVM"):
            continue  # takes too long to run,  # clf_name == "DBSCAN"

        elif clf_name == 'DBSCAN':

            # initialise classifier
            clf, init_time = call_init_classifier(clf_name)

            # fit classifier
            labels, train_time = call_fit(clf, clf_name, predict_data)
            scores = None
            predict_time = 0
        else:
            # initialise classifier
            clf, init_time = call_init_classifier(clf_name)

            # fit classifier
            clf, train_time = call_fit(clf, clf_name, train_data)

            # predict using classifier to get labels and scores
            labels, scores, predict_time = call_predict(
                clf,
                clf_name, predict_data)

        # all these three need to be set
        total_time = init_time + train_time + predict_time

        print("total run time: {}".format(total_time))

        predict_labels = predict_labels.flatten()
        labels = labels.astype(int)
        predict_labels = predict_labels.astype(int)

        # produce the reports
        # -------------------
        cls_report_dic = classification_report(
            predict_labels, labels,
            target_names=['normal', 'abnormal'], output_dict=True)

        if clf_name == 'DBSCAN':
            # show binary label version of auc score
            auc_score = roc_auc_score(predict_labels, labels)
        else:
            auc_score = roc_auc_score(predict_labels, scores)

        # collect all the metrics
        metrics_temp = pd.DataFrame({
            'Dataset': [dataset_name],
            'Classifier': [clf_name],
            'Precision': [cls_report_dic['abnormal']['precision']],
            'Recall': [cls_report_dic['abnormal']['recall']],
            'F1': [cls_report_dic['abnormal']['f1-score']],
            'Support': [cls_report_dic['abnormal']['support']],
            'AUC': [auc_score],
            'Init. time': [init_time],
            'Train time': [train_time],
            'Predict time': [predict_time],
            'Runtime': [total_time],
        })

        metrics_df = pd.concat([metrics_df, metrics_temp]
                               ).reset_index(drop=True)

    # remove the first row due to timing issues
    metrics_df = metrics_df.iloc[1:]

    return metrics_df


def format_df(metrics_df):
    # round required columns to decimal places, leave time as is
    list_round_2 = ['Precision', 'Recall', 'F1', 'AUC']
    list_round_5 = ['Init. time', 'Train time', 'Predict time', 'Runtime']
    metrics_df[list_round_2] = metrics_df[list_round_2].round(2)
    metrics_df[list_round_5] = metrics_df[list_round_5].round(5)

    # drop columns not required
    columns_to_drop = ['Dataset', 'Init. time',
                       'Train time', 'Predict time', 'Support']
    metrics_df_dropped = metrics_df.drop(columns_to_drop, axis=1)
    df = metrics_df_dropped.copy()

    return df


def get_file_data(base_path, file_name):

    # join paths to get file to path
    file_path = os.path.join(base_path, file_name)

    if file_name.endswith((".mat")):

        dataset_name = os.path.splitext(file_name)[0]

        # read in the file into numpy ndarray
        data = loadmat(file_path)

        # get the data and show
        X_original = data['X']
        y = data['y']

    elif file_name.endswith((".csv")):

        dataset_name = os.path.splitext(file_name)[0]

        df_temp = pd.read_csv(file_path, header=None)

        # replace label column with 0's and 1's
        df_temp.iloc[:, -1] = np.where(df_temp.iloc[:, -1] == 'n', 0, 1)

        # read data in as numpy arrays
        X_original = df_temp.iloc[:, 0:-1].values

        y = df_temp.iloc[:, -1].values
        y = y.reshape(y.shape[0], -1)

    elif file_name.endswith((".matv7")):

        dataset_name = os.path.splitext(file_name)[0]

        arrays = {}
        f = h5py.File(file_path)
        for k, v in f.items():
            arrays[k] = np.array(v)

        X_original = arrays['X'].T
        y = arrays['y'].T

    else:
        # do nothing as file formatted not supported
        dataset_name = None
        X_original = None
        y = None

    return dataset_name, X_original, y


# highlight bold for latex table use
def highlight_max_min(df_, col_show_max):

    df = df_.copy()
    # produce print out for paper
    for col, v in col_show_max.items():

        if v:
            # highlight maximum of columns
            maximum = df[col].max()
            bolded = df[col].apply(lambda x: "\\textbf{%s}" % x)
            df[col] = df[col].where(df[col] != maximum, bolded)
        else:
            # highlight minimum of columns
            minimum = df[col].min()
            bolded = df[col].apply(lambda x: "\\textbf{%s}" % x)
            df[col] = df[col].where(df[col] != minimum, bolded)

    return df
