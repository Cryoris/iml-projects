# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:10:08 2018

@author: jonas
"""

file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task3/'



import pandas as pd
train = pd.read_hdf(file_path+"train.h5", "train")
test = pd.read_hdf(file_path+"test.h5", "test")

#%%

train
