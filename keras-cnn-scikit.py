#-*- coding: utf-8 -*-  

from __future__ import print_function

import sys  
import os  
import time  
from sklearn import metrics  
import numpy as np  
import cPickle as pickle  
import cPickle
import numpy


import theano
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np


reload(sys)  
sys.setdefaultencoding('utf8')  

def load_data(dataset):
    data = np.empty((2960,1,64,64),dtype="float32")
    label = np.empty((2960,),dtype="uint8")
    read_file=open(dataset,'rb')  
    faces=cPickle.load(read_file)    #faces:2960*(64*64)
    labels=cPickle.load(read_file)    
    read_file.close()   
    label=labels
    num = len(faces)


    for i in range(num):
        face=np.array(faces[i].reshape(64,64))
        data[i,:,:,:] = face
        data /= np.max(data)
        data -= np.mean(data)
    return data,label
  
# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model  
  
# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print (para, val)  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  


if __name__ == "__main__":
    thresh = 0.5  
    model_save_file = None  
    model_save = {}  

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']  
    classifiers = {'NB':naive_bayes_classifier,   
                  'KNN':knn_classifier,  
                   'LR':logistic_regression_classifier,  
                   'RF':random_forest_classifier,  
                   'DT':decision_tree_classifier,  
                  'SVM':svm_classifier,  
                'SVMCV':svm_cross_validation,  
                 'GBDT':gradient_boosting_classifier  
    }  

    #load data
    data, label = load_data('D:\\CK\\data.pkl')
    #shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    
    (traindata,testdata) = (data[0:1776],data[1776:])
    (trainlabel,testlabel) = (label[0:1776],label[1776:])
    #use origin_model to predict testdata
    origin_model = cPickle.load(open("model.pkl","rb"))
    #print(origin_model.layers)
    pred_testlabel = origin_model.predict_classes(testdata,batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(" Origin_model Accuracy:",accuracy)

    #define theano funtion to get output of FC layer
    get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[10].output,allow_input_downcast=False)
    feature = get_feature(data)
    #train svm using FC-layer feature
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
   
    print ('reading training and testing data...')
    train_x = feature[0:1776]
    train_y = label[0:1776]
    test_x = feature[1776:]
    test_y = label[1776:]  
    num_train, num_feat = train_x.shape  
    num_test, num_feat = test_x.shape  
    is_binary_class = (len(np.unique(train_y)) == 2)  
    print ('******************** Data Info *********************')
    print ('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)) 
      
    for classifier in test_classifiers:  
        print ('******************* %s ********************' % classifier)  
        start_time = time.time()  
        model = classifiers[classifier](train_x, train_y)  
        print ('training took %fs!' % (time.time() - start_time))  
        predict = model.predict(test_x)  
        if model_save_file != None:  
            model_save[classifier] = model  
        if is_binary_class:  
            precision = metrics.precision_score(test_y, predict)  
            recall = metrics.recall_score(test_y, predict)  
            print ('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))  
        accuracy = metrics.accuracy_score(test_y, predict)  
        print ('accuracy: %.2f%%' % (100 * accuracy))   
  
    if model_save_file != None:  
        pickle.dump(model_save, open(model_save_file, 'wb'))

