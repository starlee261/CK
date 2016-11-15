#coding:utf-8

#相比较keras-cnn改进了激活函数，训练比例等

#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
import random,cPickle
from keras.callbacks import EarlyStopping
import numpy as np

np.random.seed(1024)  # for reproducibility


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


#加载数据
data, label = load_data('D:\\CK\\data.pkl')


#label为0~7共8个类别，keras要求形式为binary class matrices,转化一下，直接调用keras提供的这个函数
nb_class = 8
label = np_utils.to_categorical(label, nb_class)


def create_model():
	model = Sequential()
	model.add(Convolution2D(4, 9, 9, border_mode='valid',input_shape=(1,64,64))) 
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(8,5, 5, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(16,5, 5, border_mode='valid')) 
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, init='normal'))
	model.add(Activation('relu'))

	model.add(Dense(nb_class, init='normal'))
	model.add(Activation('softmax'))
	return model


#############
#开始训练模型
##############
model = create_model()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
(X_train,X_val) = (data[0:1776],data[1776:])
(Y_train,Y_val) = (label[0:1776],label[1776:])

#使用early stopping返回最佳epoch对应的model
early_stopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, batch_size=50,validation_data=(X_val, Y_val),nb_epoch=100,callbacks=[early_stopping])
cPickle.dump(model,open("D:\\CK\\model.pkl","wb"))
