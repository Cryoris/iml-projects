import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd

test = "test.csv"
train = "train.csv"
n = 1000

df = pd.read_csv(train)
Y = df['y']
X = df.iloc[:,2:]


lr = linear_model.LinearRegression()
lr.fit(X, Y)

dtest = pd.read_csv(test)
Xpred = dtest.iloc[:,1:]
Ypred = lr.predict(Xpred)

dres = pd.DataFrame({'Id' : dtest['Id'], 'y' : Ypred})
dres.to_csv("pred.csv", index=False)
print(dres)
