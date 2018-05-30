# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:10:08 2018

@author: jonas
"""

#file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task4//ladder-master/data/'
file_path = ""

import pandas as pd
train_labeled = pd.read_hdf(file_path+"train_labeled.h5", "train")
train_unlabeled = pd.read_hdf(file_path+"train_unlabeled.h5", "train")
test = pd.read_hdf(file_path+"test.h5", "test")

header = False

#%%


train_labeled.to_csv("train_labeled.csv", index=False, header=header)
train_unlabeled.to_csv("train_unlabeled.csv", index=False, header=header)
test.to_csv("test.csv", index=False, header=header)

#%%
train_labeled.drop('y', axis=1).to_csv("train_labeled_withoutLabel.csv", index=False, header=header)
train_labeled["y"].to_csv("train_labeled_justLabel.csv", index=False, header=header)

#%% Split train into train/test to be able to show accuracy

lenTest = 100

train_labeled.iloc[:-lenTest, 1:].to_csv("train_labeled_withoutLabel_useTrain.csv", index=False, header=header)
train_labeled.iloc[-lenTest:,1:].to_csv("train_labeled_withoutLabel_useTest.csv", index=False, header=header)

train_labeled["y"].iloc[:-lenTest].to_csv("train_labeled_justLabel_useTrain.csv", index=False, header=header)
train_labeled["y"].iloc[-lenTest:].to_csv("train_labeled_justLabel_useTest.csv", index=False, header=header)

