# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:15:07 2018

@author: jonas
"""

import pandas
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, RandomTreesEmbedding
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, SelectKBest, f_classif, SelectFromModel
#%%
if __name__ == '__main__':
    print "loading dataset"
    file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2/train.csv'
    input_col = ['y', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']
    dataset = pandas.read_csv(file_path, usecols = input_col)
    # print dataset.shape
    # print(dataset.head(20))
    # print(dataset.describe())
    # print(dataset.groupby('y').size())
    # dataset.plot(kind = 'box', subplots=True, layout=(4,4), sharex=False,sharey=False)
    # plt.show()
    # dataset.hist()
    # plt.show()
    # scatter_matrix(dataset)
    # plt.show()
    training_data = dataset.values
    X = training_data[:,1:len(training_data[0])]
    # X = VarianceThreshold(threshold=(0.8*(1-0.8))).fit_transform(X)
    Y = training_data[:,0]
    # X = SelectKBest(f_classif).fit_transform(X_new,Y)
    # X = SelectFromModel(ExtraTreesClassifier().fit(X_new,Y), prefit=True).transform(X_new)
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed)
    scoring = 'accuracy'
    models = []
    
#    models.append(('LR', LogisticRegression(C=0.025, class_weight='balanced', solver='newton-cg', max_iter=100, multi_class='multinomial', warm_start=True)))
#    models.append(('LDA', LinearDiscriminantAnalysis())) #not much to tweak
#    models.append(('KNN', KNeighborsClassifier(n_neighbors=8,weights='distance',algorithm='kd_tree',leaf_size=30,p=2,metric='minkowski',metric_params=None,n_jobs=1)))
#    models.append(('CART', DecisionTreeClassifier(max_depth=4)))
#    models.append(('NB', GaussianNB()))
#    for i in range(50,70,2):
#        models.append(('SVM_linear'+str(i*0.001), SVC(kernel="linear",C=i*0.001)))
#    models.append(('SVM_gamma2', SVC(gamma=2,C=1)))
#    models.append(('Gaussian', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)))
#    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=5)))
    models.append(('RandomForestClassifier', RandomForestClassifier(max_depth=7, n_estimators=136)))
    models.append(('MLPClassifier', MLPClassifier(alpha=1)))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis()))
    models.append(('RFE', RFE(estimator=SVC(kernel="linear",C=1), n_features_to_select=1, step=1)))
    models.append(('BGC', BaggingClassifier(ExtraTreesClassifier(), n_estimators = 120, bootstrap = False, n_jobs = 2)))
    models.append(('etc', ExtraTreesClassifier(n_jobs=-1, n_estimators=120, max_features=None, max_depth=10)))
    clf1 = KNeighborsClassifier(n_neighbors=8,weights='distance',algorithm='kd_tree',leaf_size=30,p=2,metric='minkowski',metric_params=None,n_jobs=1)
    clf2 = RandomForestClassifier(max_depth=5, n_estimators=136)
    clf3 = ExtraTreesClassifier(n_estimators=125, max_features=None)
    models.append(('VC', VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('etc', clf3)], voting='soft')))
    models.append(('rte', RandomTreesEmbedding()))
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print "accuracy_score:"
        print(accuracy_score(Y_validation,predictions))
        print(classification_report(Y_validation,predictions))
        # fig = plt.figure()
        # fig.suptitle('Algorithm Comparison')
        # ax = fig.add_subplot(111)
        # plt.boxplot(results)
        # ax.set_xticklabels(names)
        # plt.show()
        # knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',leaf_size=30,p=2,metric='minkowski',metric_params=None,n_jobs=1)
        # knn.fit(X_train, Y_train)
        # predictions = knn.predict(X_validation)
        # rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10, n_jobs = 5)
        # rfc.fit(X_train, Y_train)
        # bgc = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
        # bgc.fit(X_train, Y_train)
        # predictions = bgc.predict(X_validation)
        # print "accuracy_score:"
        # print(accuracy_score(Y_validation,predictions))
        # # print confusion_matrix(Y_validation,predictions)
        # print(classification_report(Y_validation,predictions))
        
        
#%%

# BGC won

# 

def BGC():
    print "loading datasets"
    train_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2/train.csv'
    input_col = ['y', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']
    trainset = pandas.read_csv(train_path, usecols = input_col)
    training_data = trainset.values
    X_train = training_data[:,1:len(training_data[0])]
    Y_train = training_data[:,0]
    
    test_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2/test.csv'
    testset = pandas.read_csv(test_path, usecols = input_col[1:])
    X_test = testset.values
    
    seed = 7
    model = BaggingClassifier(ExtraTreesClassifier(), n_estimators = 120, bootstrap = False, n_jobs = 2)
    scoring = 'accuracy'
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('BGC', cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions
    
predictions = BGC()
    
#%%
import numpy as np
ids = np.arange(0,3000)+2000
BGCpred = pandas.DataFrame(data=predictions,index=ids,columns=["y"])  
file_path = 'C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2/BGCpython.csv'
BGCpred.to_csv(path_or_buf=file_path, columns=None, header=True, index=True, )