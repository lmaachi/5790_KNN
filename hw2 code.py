#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:17:16 2023

@author: lulumaachi
"""

import numpy as np
import pandas as pd

spam_test = pd.read_csv(r'/Users/lulumaachi/Downloads/HW2_dataset/spam_test.csv')
spam_train = pd.read_csv(r'/Users/lulumaachi/Downloads/HW2_dataset/spam_train.csv')

features = ['f' + str(i) for i in range(1, 58)]
k_values = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]

def knn_prediction_without_norm(df_name):
    pred = {}
    
    global test_with_pred
    
    test_mat = spam_test[features].values
    train_mat = spam_train[features].values

    for i in range(spam_test.shape[0]):
        diff = test_mat[i, :] - train_mat
        sum_sq = np.sum(np.square(diff), axis=1)
        df = pd.DataFrame({'sum_sq': sum_sq, 'class': spam_train['class']})
        sorted_class = df.sort_values('sum_sq')['class']
        pred[i] = [sorted_class[:k].mean() > 0.5 for k in k_values]

    test_with_pred = pd.concat([spam_test, pd.DataFrame(pred, index=k_values).T.astype(int)], axis=1)

    accuracy = pd.Series({k: (test_with_pred['Label'] == test_with_pred[k]).mean() for k in k_values})
    return accuracy

accuracy = knn_prediction_without_norm(spam_test)
print("The accuracy for k =", k_values, "without normalization is\n", accuracy)

def z_score_norm(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def knn_prediction_with_norm():
    pred = {}
    
    global test_with_pred
    
    train_mean = spam_train[features].mean().values
    train_std_dev = spam_train[features].std().values

    train_mat_normalized = (spam_train[features].values - train_mean) / train_std_dev
    test_mat_normalized = (spam_test[features].values - train_mean) / train_std_dev

    for i in range(spam_test.shape[0]):
        diff = test_mat_normalized[i, :] - train_mat_normalized
        sum_sq = np.sum(np.square(diff), axis=1)
        sorted_indices = np.argsort(sum_sq)
        sorted_class = spam_train['class'].iloc[sorted_indices].values
        pred[i] = [np.mean(sorted_class[:k]) > 0.5 for k in k_values]

    test_with_pred = pd.concat([spam_test, pd.DataFrame(pred, index=k_values).T.astype(int)], axis=1)
    accuracy_normalized = pd.Series({k: (test_with_pred['Label'] == test_with_pred[k]).mean() for k in k_values})
    
    return accuracy_normalized

accuracy_normalized = knn_prediction_with_norm()
print("The accuracy for k =", k_values, "with z-score normalization is\n", accuracy_normalized)

first_50_instances = test_with_pred.iloc[:50]

label_mapping = {1: 'spam', 0: 'no'}

result_table = pd.DataFrame(index=range(0, 50))  

for k in k_values:
    result_table[f'K={k}'] = first_50_instances[k].map(label_mapping)

print(result_table)