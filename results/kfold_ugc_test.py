# -*- coding: utf-8 -*-
"""
This script shows how to apply k-folds train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Zhengzhong Tu
"""
import warnings
import time
import os
# ignore all warnings
warnings.filterwarnings("ignore")
# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

import seaborn as sn

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================'''

model_name = 'SVR'
data_name = 'YOUTUBE_UGC_480P'
algo_name = 'VFSA'
color_only = False
save_path = 'model'
print("Evaluating algorithm {} with {} on dataset {} ...".format(algo_name, model_name, data_name))

## read YOUTUBE_UGC
# csv_file = './mos_files/'+data_name+'_metadata.csv'
csv_file = '/mnt/storage/home/um20242/scratch/RAPIQUE-main/mos_files/YOUTUBE_UGC_480P_metadata.csv'
npy_file = np.load('/mnt/storage/home/um20242/scratch/VSFA-master/CNN_features_360P/0_resnet-50_res5c.npy',allow_pickle=True)
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values
# y3 = array[1:,4]
y3 = array[1:,5]
y3 = np.array(list(y3), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
# print(X_mat)
X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)
# print(X3)

X = np.vstack((X3))
y = np.vstack((y3.reshape(-1,1))).squeeze()

## preprocessing
from sklearn.impute import SimpleImputer
X[np.isinf(X)] = np.nan
imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
X = imp.transform(X)

## parameter search on k-fold
param_grid = {'C': np.logspace(1, 10, 10, base=2),
              'gamma': np.logspace(-8, 1, 10, base=2)}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=4, verbose=2)

## scaler
scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)
print(len(X))
print(len(y))
# grid search
grid.fit(X, y)
best_params = grid.best_params_

## finalize SVR model on the combined set
C = best_params['C']
gamma = best_params['gamma']
model = SVR(kernel='rbf', gamma=gamma, C=C)
# Standard min-max normalization of features

# Fit training set to the regression model
model.fit(X, y)

# Apply scaling
y_pred = model.predict(X)

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
# logistic regression
try:
    beta = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, pcov = curve_fit(logistic_func, y_pred, \
        y, p0=beta, maxfev=100000000)
except:
    raise Exception('Fitting logistic function time-out!!')
y_pred_logistic = logistic_func(y_pred, *popt)
print('======================================================')
# print('mos:', str(y3))
# print('mos:', str(y))
# print('y_pred', y_pred)
# print('y_pred_logistic', y_pred_logistic)

# 画图
data = {'MOS': y,
        'y_pred': y_pred,
        'Predicted Score': y_pred_logistic}
d = pandas.DataFrame(data)
print(d)
fig = sn.regplot(x='Predicted Score', y='MOS', data=d, marker='o',
                order = 2,  # 默认为1，越大越弯曲
                scatter_kws={'color': '#016392', },  # 设置散点属性，参考plt.scatter
                line_kws={'linestyle': '--', 'color': '#c72e29'}  # 设置线属性，参考 plt.plot
                )
plt.show()
plt.title("360P YT-UGC", fontsize=10)
reg_fig = fig.get_figure()
fig_path = '/mnt/storage/home/um20242/scratch/VSFA-master/figs'
reg_fig.savefig(fig_path + algo_name + '_' + data_name, dpi=400)

plcc = scipy.stats.pearsonr(y, y_pred_logistic)[0]
rmse = np.sqrt(mean_squared_error(y, y_pred_logistic))
srcc = scipy.stats.spearmanr(y, y_pred_logistic)[0]
krcc = scipy.stats.kendalltau(y, y_pred_logistic)[0]

# print results for each iteration
print('======================================================')
print('Best results in CV grid search')
print('SRCC: ', srcc)
print('KRCC: ', krcc)
print('PLCC: ', plcc)
print('RMSE: ', rmse)
print('======================================================')

print('model:', model)
print('scaler:', scaler)
print('popt:', popt)
joblib.dump(model, os.path.join(save_path,algo_name+'_trained_svr.pkl'))
joblib.dump(scaler, os.path.join(save_path,algo_name+'_trained_scaler.pkl'))
scipy.io.savemat(os.path.join(save_path,algo_name+'_logistic_pars.mat'), \
    mdict={'popt': np.asarray(popt, dtype=np.float)})
