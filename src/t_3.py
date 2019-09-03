'''
 * @desc : cnn训练mnist
 * @auth : TYF
 * @date : 2019/8/31 - 15:58
'''


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time

#数据load
mnist = input_data.read_data_sets('C:/work/dataSet/mnist/',one_hot=True)

print('---------------------样本个数--------------------------')
print('train:',mnist.train.num_examples)
print('validation:',mnist.validation.num_examples)
print('test:',mnist.test.num_examples)
print('-----------------------------------------------')
print('train images:',mnist.train.images.shape)
print('validation images:',mnist.validation.images.shape)
print('test images:',mnist.test.images.shape)
print('---------------------样本形状--------------------------')
print('shape 0:',mnist.train.images[0].shape)  #(784,) 单个图片样本已经转为1行且标准化
print('shape 0:',mnist.train.images[0])


#权重(过滤器)建立函数
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W') #truncated_normal函数产生截断正态分布随机数初始化权重

#偏差建立函数
def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='B') #常数0.1初始化偏差张量

#卷积计算函数 输入28x28,输出28x28
def conv2d(x,W):
    return tf.nn.conv2d(x,W,padding='SAME',strides=[1,1,1,1])
    #x:输入参数,必须为4维张量
    #W:滤镜权重
    #strdes:步长[1,s,s,1],滤镜每次移动时从左到右一步,从上到下一步
    #SAME:边界补0,让输入和输出大小不变

#池化(缩减采样) 输入28x28,输出14x14
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #x:输入
    #ksize:采样窗口大小[1,h,w,1]
    #strides:采样窗口移动跨度[1,s,s,1],移动是每次从左到右两步,从上到下两步

#建立模型
x = tf.placeholder(shape=[None,784],name='x',dtype='float') #1x784
x_image = tf.reshape(x,[-1,28,28,1]) #1x784转28x28
#卷积1
W1 = weight([5,5,1,16])  #16个 尺寸为5x5的卷积核
B1 = bias([16]) #偏差
Conv1 = conv2d(x_image,W1)+B1
c1_Conv = tf.nn.relu(Conv1) #得到16层 28x28
#池化1
c1_Pool = max_pool_2x2(c1_Conv) #得到16层 14x14
#卷积2
W2 = weight([1,5,5,32]) #32个 5x5的卷积核
B2 = bias([32])  #偏差
Conv2 = conv2d(c1_Pool,W2)+B2
c2_Conv = tf.nn.relu(Conv2) #得到32层 14x14
#池化2
c2_Pool = max_pool_2x2(c2_Conv) #得到36层 7x7
#展平
d_Flat = tf.reshape(c2_Pool,[-1,36*7*7])
#隐层
W3 = weight([36*7*7,128]) #36*7*7个神经元连接到128个神经元
B3 = bias([128]) #偏差
d_Hidden = tf.nn.relu(tf.matmul(d_Flat,W3)+B3) #全连接计算
#DropOut层
d_Hidden_Dropout = tf.nn.dropout(d_Hidden,keep_prob=0.8)
#输出层
W4 = weight([128,10])
B4 = bias([10])
d_Out = tf.nn.relu(tf.matmul(d_Hidden_Dropout,W4)+B4)

