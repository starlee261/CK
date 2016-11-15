#-*- coding: utf-8 -*- 
	
import numpy
import pylab  
#读取data.pkl文件，分为训练集（2072个样本），验证集（444个样本），测试集（444个样本）  
import cPickle  
import random
read_file=open('D:\\CK\\data.pkl','rb')  
faces=cPickle.load(read_file)    
label=cPickle.load(read_file)    
read_file.close()   
  
train_data=numpy.empty((2072,4096))  
train_label=numpy.empty(2072)  
valid_data=numpy.empty((444,4096))  
valid_label=numpy.empty(444)  
test_data=numpy.empty((444,4096))  
test_label=numpy.empty(444)  

li=range(2960)
random.shuffle(li)

# print li[1]
# train_data[1]=faces[li[1]]
# train_label[1]=label[li[1]]  

#测试是否正确
# img=train_data[1].reshape(64,64)
# pylab.imshow(img)  
# pylab.gray()  
# pylab.show() 
# print train_label[1]
  
for i in range(2960):
	if (i<2072):
		train_data[i]=faces[li[i]]
    	train_label[i]=label[li[i]]  
    elif (i>=2072 and i<2516):
        valid_data[i-2072]=faces[li[i]]  
   		valid_label[i-2072]=label[li[i]]
   	else:  
    	test_data[i-2516]=faces[li[i]]  
    	test_label[i-2516]=label[li[i]] 


#另外一种实现
# random.seed(1)
# random.shuffle(faces)
# random.seed(1)
# random.shuffle(label)