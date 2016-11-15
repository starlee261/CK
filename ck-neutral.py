#-*- coding: utf-8 -*- 
import os
import numpy
from PIL import Image	
from pylab import *		




def get_imlist(path):	#此函数读取特定文件夹下的png格式图像

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]


neutral=get_imlist(r"D:\\CK\\0")	#r""是防止字符串转译
#print neutral 	#这里以list形式输出bmp格式的所有图像（带路径）
d=len(neutral)	#这可以以输出图像个数


data=numpy.empty((d,64*64))	#建立d*（64*64）的矩阵
while d>0:
	img=Image.open(neutral[d-1])	#打开图像
	#img_ndarray=numpy.asarray(img)
	img_ndarray=numpy.asarray(img,dtype='float64')/256	#将图像转化为数组并将像素转化到0-1之间
	data[d-1]=numpy.ndarray.flatten(img_ndarray)	#将图像的矩阵形式转化为一维数组保存到data中
	d=d-1
print len(data)
print shape(data)[1]	#输出矩阵大小
