# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:10:08 2018

@author: jonas
"""

<<<<<<< HEAD
#file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task4//ladder-master/data/'
file_path = ""
=======
file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task4/'


>>>>>>> 41e0f7da12789f966221c8a377a87d6e07c02ed6

import pandas as pd
train_labeled = pd.read_hdf(file_path+"train_labeled.h5", "train")
train_unlabeled = pd.read_hdf(file_path+"train_unlabeled.h5", "train")
test = pd.read_hdf(file_path+"test.h5", "test")

<<<<<<< HEAD
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

=======
#%%

train_labeled.to_csv("train_labeled.csv", index=False)
train_unlabeled.to_csv("train_unlabeled.csv", index=False)
test.to_csv("test.csv", index=False)
>>>>>>> 41e0f7da12789f966221c8a377a87d6e07c02ed6
