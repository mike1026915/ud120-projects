#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
sys.path.insert(0,'/usr/lib/python2.7/site-packages/')
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn import svm

#features_tirain = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
for c in (10000,):
    clf = svm.SVC(kernel='rbf',C=c)
    clf.fit(features_train, labels_train)

#t0 = time()
    pred = clf.predict(features_test)
    print len(pred)
    print len([p for p in pred if p == 1])
#print time()  t0
#print pred

    #print clf.score(features_test, labels_test)

