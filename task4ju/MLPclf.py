import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train = pd.read_hdf("data/train.h5", "train")
test = pd.read_hdf("data/test.h5", "test")
sample = pd.read_csv("data/sample.csv")

y = np.array(train['y'])
X_train = np.array(train.iloc[:, 2:])
X_test = np.array(test.iloc[:,1:])



seed = 42.
scoring = 'accuracy'
Kfold = KFold(n_splits=10, random_state=seed)

# preprocessing
print('preprocessing...')
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# results from validation
s = 70
n, m = 12, 8 #or 12, 8
phi = 'tanh'
eta = 'constant' # all methods achieved same score
a = 0.01 # keep as high as possible?

#for m in 2*np.arange(1, 5):
# for n in 3*np.arange(1, 5):
#for a in 0.1**np.arange(1, 6):
#for phi in ['logistic', 'tanh', 'relu']:
#for s in 10*np.arange(1,10):
for dummy in [1]:
    print(a)
    ch2 = SelectKBest(chi2, k=s)
    X_ = ch2.fit_transform(X_train, y)

    print('fitting...')
    clf = MLPClassifier(solver='adam', activation = phi, alpha=a, hidden_layer_sizes=(n,m), random_state=1, learning_rate=eta)
    
    cv_results = cross_val_score(clf, X_, y, cv=Kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('MLPClassifier', cv_results.mean(), cv_results.std())
    print(msg)
    
    clf.fit(X_, y)
    print("predicting...")
    X_test = ch2.transform(X_test)
    y_pred = clf.predict(X_test)



    # evaluation
    #acc = accuracy_score(y, y_pred)


    # write predict file
    pred = pd.DataFrame({'Id' : sample['Id'], 'y' : y_pred})
    pred.to_csv("pred.csv", index=False)
