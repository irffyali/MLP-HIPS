# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:02:25 2019

@author: irffy
"""

import pandas as pd
from pandas.compat import StringIO
import re
import numpy as np 
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('HIPSdata.csv')
data['parsec'] = 1000/data['Plx'] 
data['T'] = 5601/((data['B-V'] +0.4)**(2/3))
data['L'] = (15-data['Vmag']-5*np.log10(data['Plx']))/2.5
data['Lacc'] = (10**data['L']) #luminosity in no. of solar luminosityes
def condition(x):
    if x < 3500:
        return 'M'
    elif 3500< x < 5000:
        return 'K'
    elif 5000< x  < 6000:
        return 'G'  
    elif 6000< x < 7500:
        return 'F'
    elif 7500< x < 11000:
        return 'A'
    elif 11000< x < 25000:
        return 'B'
plt.scatter( data['T'],data['L'])
ax = plt.gca()
ax.invert_xaxis() #hertzprung russel diagram
data['class'] = data['T'].apply(condition)
def condition2(x):
    if x == 'M':
        return 1
    elif x == 'K':
        return 2 
    elif x == 'G':
        return 3 
    elif x == 'F':
        return 4 
    elif x == 'A':
        return 5 
    elif x == 'B':
        return 6 
data['classvar'] = data['class'].apply(condition2)  
data = data.dropna()



X = data[['B-V', 'L']].values #color index and Luminosity
y = data['classvar'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #split dataset into training and testing data

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sun = np.array([[5778, 1]])
sunpred = sc.transform(sun)

mlp = MLPClassifier(hidden_layer_sizes=(2,2))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("svm Accuracy:",metrics.accuracy_score(y_test, y_pred)) #99.8% overfitted?
print("mlp Accuracy:",metrics.accuracy_score(y_test, predictions)) 

#using models to predict the class of the sun which is G i.e. 3
sun = np.array([[0.656, np.log10(1)]])
sunpred = sc.transform(sun)
print(mlp.predict(sunpred))
print(clf.predict(sunpred))



#both models predict the class of the sun correctly.



