#-*- coding: utf-8 -*- 
import cPickle  
import pylab  
from pylab import *		#导入savetxt模块

read_file=open('D:\\CK\\data.pkl','rb')    
faces=cPickle.load(read_file)  
read_file.close()   
img=faces[0].reshape(64,64)
savetxt('faces.txt',img,fmt="%.0f") #将矩阵保存到txt文件中
pylab.imshow(img)  
pylab.gray()  
pylab.show() 