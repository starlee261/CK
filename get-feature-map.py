#-*- coding: utf-8 -*-  
from __future__ import print_function
import cPickle,theano
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

#load the saved model
model = cPickle.load(open("model.pkl","rb"))

#define theano funtion to get output of  FC layer
get_feature = theano.function([model.layers[0].input],model.layers[11].output,allow_input_downcast=False) 

#define theano funtion to get output of  first Conv layer 
get_featuremap = theano.function([model.layers[0].input],model.layers[2].output,allow_input_downcast=False) 


data, label = load_data('D:\\CK\\data.pkl')

# visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images
feature = get_feature(data[0:10])  #visualize these images's FC-layer feature
plt.imshow(feature,cmap = cm.Greys_r)
plt.show()

#visualize feature map of Convolution Layer
num_fmap = 4	#number of feature map
for i in range(num_fmap):
	featuremap = get_featuremap(data[0:10])
	plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
	plt.show()