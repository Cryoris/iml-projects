# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:10:08 2018

@author: jonas
"""

file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task4/'



import pandas as pd
train_labeled = pd.read_hdf(file_path+"train_labeled.h5", "train")
train_unlabeled = pd.read_hdf(file_path+"train_unlabeled.h5", "train")
test = pd.read_hdf(file_path+"test.h5", "test")

#%%

train_labeled.to_csv("train_labeled.csv", index=False)
train_unlabeled.to_csv("train_unlabeled.csv", index=False)
test.to_csv("test.csv", index=False)
