#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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

print features_train
print features_train[0]
print len(features_train[0])


#########################################################
### your code goes here ###
#########################################################

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,min_samples_split=40)
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)
