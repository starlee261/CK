#-*- coding: utf-8 -*- 
import os
import numpy
import pylab 
from PIL import Image	#导入Image模块
from pylab import *		#导入savetxt模块

img=Image.open("D:\\CK\\0\\S005_001_00000001_64.png")
img_ndarray=numpy.asarray(img,dtype='float64')
savetxt('xiangsu.txt',img_ndarray,fmt="%.0f") #将矩阵保存到txt文件中
pylab.imshow(img_ndarray)  
pylab.gray()  
pylab.show() 