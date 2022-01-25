#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 00:16:04 2022

@author: Nassir, Mohammad
"""

from utilities import apply_methods
from utilities import plot_data, show_anomalies
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


image_save_path = (
    '/Users/nassirmohammad/Google Drive/docs/'
    'Anomaly_detection_using_principles_of_human_perception/'
    'figures/')
image_save_switch = False

# =============================================================================
# NAB Data Corpus - real known cause
# =============================================================================

#################################
#
# Machine temperature system failure
#
#################################

# %% read in the data
machine_temp_sys_fail_path = "../data/machine_temperature_system_failure.csv"

machine_df = pd.read_csv(machine_temp_sys_fail_path,
                         engine='c',
                         parse_dates=['timestamp'])
print(machine_df.info())

# %% plot the data
plot_data(machine_df['value'],
          save_path=image_save_path + '',
          save_switch=False,
          balls_height=100)

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(x="timestamp", y="value", data=machine_df, ci=None)

# set ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
# rotate the labels
ax.tick_params(axis='x', rotation=45)
# set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# %% run outlier detectors
vals = np.array(machine_df['value'])
results = apply_methods(vals)

# %% plot anomalies results
show_anomalies(machine_df, results, 'label_perception',
               alg_name='perception_method')

show_anomalies(machine_df, results, 'label_z_3',
               alg_name='z_score_3_method')

# %% assign labels to each data value using method output
label_names = ['label_perception', 'label_z_2.5',
               'label_z_3', 'label_mad', 'label_iqr']
method_names = ['perception_method', 'z_score_2_5_method',
                'z_score_3_method', 'mad_method', 'iqr_method']

for algorithm_label, alg_name in zip(label_names, method_names):
    machine_df[algorithm_label] = (
        machine_df['value'].isin(results.loc[alg_name])).astype(int)

# group by each method on binary class, and take max of anomalies
column_name = 'value'

perception_threshold_max = machine_df.groupby('label_perception')[
    column_name].max()
z_threshold_25_max = machine_df.groupby('label_z_2.5')[column_name].max()
z_threshold_3_max = machine_df.groupby('label_z_3')[column_name].max()
mad_threshold_max = machine_df.groupby('label_mad')[column_name].max()
box_threshold_max = machine_df.groupby('label_iqr')[column_name].max()

# %% plot the output data
plt.rcParams["font.size"] = "14"
fig, ax = plt.subplots(figsize=(16, 8))

ax = sns.lineplot(x="timestamp", y="value", data=machine_df, ci=None)

# rotate the labels
ax.tick_params(axis='x', rotation=45)

# set ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator())

# set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# draw the method decision boundary
ax.axhline(y=perception_threshold_max[1], linewidth=2,
           color='black', linestyle='--', label='perception')
ax.axhline(y=z_threshold_3_max[1], linewidth=2,
           color='purple', linestyle='-', label='z-score')
ax.axhline(y=mad_threshold_max[1], linewidth=2,
           color='orange', linestyle='-.', label='modified z-score')
ax.axhline(y=box_threshold_max[1], linewidth=2,
           color='green', linestyle=':', label='IQR')

ax.legend(loc='lower left')
# plt.title("Industrial machine temperature", fontsize=18);
plt.xlabel('Timestamp', fontsize=15)
plt.ylabel('Temperature', fontsize=15)

if image_save_switch is True:
    plt.savefig(image_save_path + 'machine_temperature.png')

# draw the second output plot
plt.rcParams["font.size"] = "14"

fig, ax = plt.subplots(figsize=(16, 8))

ax = sns.histplot(machine_df['value'], kde=False,
                  color='gainsboro', edgecolor='gainsboro')

# ax.axvline(x=z_threshold_25_max[1], ymax=0.1,
#           linewidth=2, color='red', label='Z-score 2.5')
ax.axvline(x=perception_threshold_max[1], ymax=0.1, linewidth=2,
           color='black', linestyle='--', label='perception')
ax.axvline(x=z_threshold_3_max[1], ymax=0.1, linewidth=2,
           color='purple', linestyle='-', label='z-score')
ax.axvline(x=mad_threshold_max[1], ymax=0.1, linewidth=2,
           color='orange', linestyle='-.', label='modified z-score')
ax.axvline(x=box_threshold_max[1], ymax=0.1, linewidth=2,
           color='green', linestyle=':', label='IQR')

# ax.axvline(x=np.median(machine_df['value']), ymax=0.1,
#            linewidth=2, color='pink', label='Median')
ax.legend(loc='upper left')

# plt.title("Industrial machine temperature", fontsize=18);
plt.xlabel('Temperature', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

if image_save_switch is True:
    plt.savefig(image_save_path + 'machine_temperature_dist.png')

#################################
#
# Ambient temperature
#
#################################

# %% read in the data
ambient_path = "../data/ambient_temperature_system_failure.csv"

ambient_df = pd.read_csv(ambient_path, engine='c', parse_dates=['timestamp'])
print(ambient_df.info())

# %% plot the data
plot_data(ambient_df['value'], balls_height=20)

fig, ax = plt.subplots(figsize=(12, 6))

ax = sns.lineplot(x="timestamp", y="value", data=ambient_df, ci=None)

# rotate the labels
ax.tick_params(axis='x', rotation=45)
# set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())
# set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# %% run outlier detectors
vals = np.array(ambient_df['value'])
results = apply_methods(vals)

# %% plot anomalies results
show_anomalies(ambient_df, results, 'label_perception',
               alg_name='perception_method')

show_anomalies(ambient_df, results, 'label_mad',
               alg_name='mad_method')


# %% assign labels to each data value using method output
label_names = ['label_perception', 'label_z_2.5',
               'label_z_3', 'label_mad', 'label_iqr']
method_names = ['perception_method', 'z_score_2_5_method',
                'z_score_3_method', 'mad_method', 'iqr_method']

for algorithm_label, alg_name in zip(label_names, method_names):
    ambient_df[algorithm_label] = (
        ambient_df['value'].isin(results.loc[alg_name])).astype(int)

column_name = 'value'

perception_upper = ambient_df[ambient_df['value'] > np.median(
    ambient_df['value'])].groupby('label_perception')[column_name].min()[1]
perception_lower = ambient_df[ambient_df['value'] < np.median(
    ambient_df['value'])].groupby('label_perception')[column_name].max()[1]

z_25_upper = ambient_df[ambient_df['value'] > np.median(
    ambient_df['value'])].groupby('label_z_2.5')[column_name].min()[1]
z_25_lower = ambient_df[ambient_df['value'] < np.median(
    ambient_df['value'])].groupby('label_z_2.5')[column_name].max()[1]

z_3_upper = ambient_df[ambient_df['value'] > np.median(
    ambient_df['value'])].groupby('label_z_3')[column_name].min()[1]
z_3_lower = ambient_df[ambient_df['value'] < np.median(
    ambient_df['value'])].groupby('label_z_3')[column_name].max()[1]

iqr_upper = ambient_df[ambient_df['value'] > np.median(
    ambient_df['value'])].groupby('label_iqr')[column_name].min()[1]
iqr_lower = ambient_df[ambient_df['value'] < np.median(
    ambient_df['value'])].groupby('label_iqr')[column_name].max()[1]

print("perception: {}, {}".format(perception_upper, perception_lower))
print("z_2_5: {}, {}".format(z_25_upper, z_25_lower))
print("z_3: {}, {}".format(z_3_upper, z_3_lower))
print("iqr: {}, {}".format(iqr_upper, iqr_lower))

# %% plot the data
plt.rcParams["font.size"] = "13"
fig, ax = plt.subplots(figsize=(18, 12))
ax = sns.lineplot(x="timestamp", y="value", data=ambient_df, ci=None)

# rotate the labels
ax.tick_params(axis='x', rotation=45)

# set ticks every week
ax.xaxis.set_major_locator(mdates.MonthLocator())

# #set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax.axhline(y=perception_upper, linewidth=2, color='black',
           linestyle='--', label='perception')

# ax.axhline(y=z_25_upper, linewidth=2, color='red', label='Z-score 2.5')
ax.axhline(y=z_3_upper, linewidth=2, color='purple',
           linestyle='-', label='z-score')
# ax.axhline(y=mad_threshold_max[1], linewidth=2, color='orange',
# label='Modified Z-score')
ax.axhline(y=iqr_upper, linewidth=2, color='green', linestyle=':', label='IQR')

ax.axhline(y=perception_lower, linewidth=2, color='black', linestyle='--',)
# ax.axhline(y=z_25_lower, linewidth=2, color='red', linestyle='-',)
ax.axhline(y=z_3_lower, linewidth=2, color='purple', linestyle='-',)
# ax.axhline(y=mad_threshold_max[1], linewidth=2, color='orange',
# label='Modified Z-score')
ax.axhline(y=iqr_lower, linewidth=2, color='green', linestyle=':',)
ax.legend()

# plt.title('Ambient office temperature', fontsize=18)
plt.xlabel('Timestamp', fontsize=15)
plt.ylabel('Temperature', fontsize=15)

if image_save_switch is True:
    plt.savefig(image_save_path + 'ambient_temperature.png')


# plot the second
plt.rcParams["font.size"] = "14"

fig, ax = plt.subplots(figsize=(16, 8))

ax = sns.histplot(ambient_df['value'], kde=False,
                  color='gainsboro', edgecolor='gainsboro')

ax.axvline(x=perception_upper, ymax=0.1, linewidth=2,
           color='black', label='perception', linestyle='--',)
# ax.axvline(x=z_25_upper, ymax = 0.1, linewidth=2, color='red', label='Z-
# score 2.5')
ax.axvline(x=z_3_upper, ymax=0.1, linewidth=2,
           color='purple', label='z-score', linestyle='-',)
ax.axvline(x=iqr_upper, ymax=0.1, linewidth=2,
           color='green', label='IQR', linestyle=':',)
# ax.axvline(x=np.median(ambient_df['value']), ymax = 0.1, linewidth=2,
# color='pink', label='Median')

ax.axvline(x=perception_lower, ymax=0.1, linewidth=2,
           color='black', linestyle='--',)
# ax.axvline(x=z_25_lower, ymax = 0.1, linewidth=2, color='red', label='Z-
# score 2.5')
ax.axvline(x=z_3_lower, ymax=0.1, linewidth=2, color='purple', linestyle='-',)
ax.axvline(x=iqr_lower, ymax=0.1, linewidth=2, color='green', linestyle=':',)
# ax.axvline(x=np.median(ambient_df['value']), ymax = 0.1, linewidth=2,
# color='pink', label='Median')

ax.legend(loc='upper right')
# plt.title('Ambient office temperature', fontsize=18)

plt.xlabel('Temperature', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

if image_save_switch is True:
    plt.savefig(image_save_path + 'ambient_temperature_dist.png')

#################################
#
# Tweets and AdExchange
#
#################################

# %% read cpu data
# read the paths
# tweets_path = \
#     "/Users/nassirmohammad/projects/data/nab/realTweets/realTweets/"
ads_exchange_path = "../data/"
# real_traffic_path =
# "/Users/nassirmohammad/projects/data/nab/realTraffic/realTraffic/"

# similar
# cpu_path = tweets_path + "Twitter_volume_IBM.csv"
cpu_path = ads_exchange_path + "exchange-4_cpm_results.csv"
# cpu_path = real_traffic_path + "TravelTime_387.csv"

# import the dataframe
cpu_df = pd.read_csv(cpu_path, engine='c', parse_dates=['timestamp'])
cpu_df.info()

# %% plot the data
plot_data(cpu_df['value'], balls_height=20)

fig, ax = plt.subplots(figsize=(12, 6))

ax = sns.lineplot(x="timestamp", y="value", data=cpu_df, ci=None)

# rotate the labels
ax.tick_params(axis='x', rotation=45)
# set ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
# set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# %% run outlier detectors
vals = np.array(cpu_df['value'])
results = apply_methods(vals)

# %% plot anomalies results
show_anomalies(cpu_df, results, 'label_perception',
               alg_name='perception_method')

show_anomalies(cpu_df, results, 'label_mad',
               alg_name='mad_method')

# %%
label_names = ['label_perception', 'label_z_2.5',
               'label_z_3', 'label_mad', 'label_iqr']
method_names = ['perception_method', 'z_score_2_5_method',
                'z_score_3_method', 'mad_method', 'iqr_method']

for algorithm_label, alg_name in zip(label_names, method_names):
    cpu_df[algorithm_label] = (cpu_df['value'].isin(
        results.loc[alg_name])).astype(int)

column_name = 'value'

perception_upper = cpu_df[cpu_df['value'] > np.median(
    cpu_df['value'])].groupby('label_perception')[column_name].min()[1]
# our_lower = cpu_df[cpu_df['value'] < np.median(cpu_df['value'])].groupby(
#     'label_our')[column_name].max()[0]

z_25_upper = cpu_df[cpu_df['value'] > np.median(cpu_df['value'])].groupby(
    'label_z_2.5')[column_name].min()[1]
# z_25_lower = cpu_df[cpu_df['value'] < np.median(cpu_df['value'])].groupby(
#     'label_z_2.5')[column_name].max()[0]

mad_upper = cpu_df[cpu_df['value'] > np.median(cpu_df['value'])].groupby(
    'label_mad')[column_name].min()[1]

z_3_upper = cpu_df[cpu_df['value'] > np.median(cpu_df['value'])].groupby(
    'label_z_3')[column_name].min()[1]
# z_3_lower = cpu_df[cpu_df['value'] < np.median(cpu_df['value'])].groupby(
#     'label_z_3')[column_name].max()[0]

iqr_upper = cpu_df[cpu_df['value'] > np.median(cpu_df['value'])].groupby(
    'label_iqr')[column_name].min()[1]
# iqr_lower = cpu_df[cpu_df['value'] < np.median(cpu_df['value'])].groupby(
#     'label_iqr')[column_name].max()[0]

print("perception: {}".format(perception_upper))
print("z_2_5: {}".format(mad_upper))
print("z_3: {}".format(z_3_upper))
print("iqr: {}".format(iqr_upper))

# %%
plt.rcParams["font.size"] = "14"
fig, ax = plt.subplots(figsize=(16, 8))
ax = sns.lineplot(x="timestamp", y="value", data=cpu_df, ci=None)

# rotate the labels
ax.tick_params(axis='x', rotation=45)

# set ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator())

# set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax.axhline(y=perception_upper, linewidth=2, color='black',
           linestyle='--', label='perception')
# ax.axhline(y=z_25_upper, linewidth=2, color='red', label='Z-score 2.5')
ax.axhline(y=z_3_upper, linewidth=2, color='purple',
           linestyle='-', label='z-score')
ax.axhline(y=mad_upper, linewidth=2, color='orange',
           linestyle='-.', label='modified z-score')
ax.axhline(y=iqr_upper, linewidth=2, color='green', linestyle=':', label='IQR')

# ax.scatter(known_cpu_df['timestamp'],known_cpu_df['value'], color='red')
ax.legend()

# plt.title('Online advertisement clicking rates', fontsize=18)
plt.xlabel('Timestamp', fontsize=15)
plt.ylabel('Rate', fontsize=15)

if image_save_switch is True:
    plt.savefig(image_save_path + 'adexchange_4.png')

# %%

plt.rcParams["font.size"] = "14"

fig, ax = plt.subplots(figsize=(16, 8))

ax = sns.histplot(cpu_df['value'], kde=False,
                  color='gainsboro', edgecolor='gainsboro')

# plt.title('Online advertisement clicking rates', fontsize=18)
plt.xlabel('Rate', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

# ax.axvline(x=z_threshold_25_max[1], ymax=0.1,
#            linewidth=2, color='red', label='Z-score 2.5')
ax.axvline(x=perception_upper, ymax=0.1, linewidth=2,
           color='black', linestyle='--', label='perception')
ax.axvline(x=z_3_upper, ymax=0.1, linewidth=2,
           color='purple', linestyle='-', label='z-score')
ax.axvline(x=mad_upper, ymax=0.1, linewidth=2, color='orange',
           linestyle='-.', label='modified z-score')
ax.axvline(x=iqr_upper, ymax=0.1, linewidth=2,
           color='green', linestyle=':', label='IQR')

# ax.axvline(x=np.median(machine_df['value']), ymax=0.1,
#            linewidth=2, color='pink', label='Median')
ax.legend(loc='upper right')

if image_save_switch is True:
    plt.savefig(image_save_path + 'adexchange_4_dist.png')
