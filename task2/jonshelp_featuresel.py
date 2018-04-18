import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LSHForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, f_classif, SelectFromModel


from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest

from sklearn.model_selection import KFold
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = np.array(train['y'])
X = np.array(train.iloc[:, 2:])

seed = 7
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#for i in 10*np.arange(6, 15):
#for i in np.arange(1, 9)/100.:
for i in np.array([0.01]):
    lsvc = LinearSVC(C=i, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_sel = model.transform(X)
    print(i)
    print(X_sel.shape)
    model = BaggingClassifier(ExtraTreesClassifier(), n_estimators = 120, bootstrap = False, n_jobs = 2)
    cv_results = model_selection.cross_val_score(model, X_sel, y, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('BGC', cv_results.mean(), cv_results.std())
    print(msg)

model.fit(X, y)
y_pred = model.predict(test.iloc[:,1:])
dres = pd.DataFrame({'Id' : test['Id'], 'y' : y_pred})
dres.to_csv("pred.csv", index=False)
print(dres)



