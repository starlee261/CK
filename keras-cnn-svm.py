#coding:utf-8

from __future__ import print_function
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np

def load_data(dataset):
    data = np.empty((2960,1,64,64),dtype="float32")
    label = np.empty((2960,),dtype="uint8")
    read_file=open(dataset,'rb')  
    faces=cPickle.load(read_file)    #faces:2960*64*64
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

def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=5)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("keras-cnn-svm Accuracy:",accuracy)

def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=8,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)

    print("keras-cnn-rf Accuracy:",accuracy)

if __name__ == "__main__":
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
    svc(feature[0:1776],label[0:1776],feature[1776:],label[1776:])
