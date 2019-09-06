'''
 * @desc : tensorflow实现 alexNet
 * @auth : TYF
 * @date : 2019/8/31 - 15:58

 AlexNet特点
 ReLU: 解决了Sigmoid在深网络中的梯度弥散问题
 Dropout: 随机忽略隐层中部分节点(类似图片随机像素点置黑相当于新创建了一个更加模糊但不影响识别的样本)
 LRN: 对局部神经元的活动创建竞争机制,影响较大的节点扩大其影响
 CNN: 使用重叠的最大池化,避免平均池化的模糊化效果,设置步长比池化核尺寸小使得池化层的输出之间会有重叠提升特征丰富性

 AlexNet一共8层,5个卷积层,3个全连接层,结构如下:

  卷积层       输入          卷积核(Ksize/channels/number/stride)       激活      输出           数据增强          池化核(Psize/stride)      最终输出
  conv1       227*227*3     11*11/  3/   64/   4*4/                    ReLU     55*55*64       LRN              3*3/  2*2/                27*27*64
  conv2       27*27*64      5*5/    64/  192/  1*1/                    ReLU     23*23*192      LRN              3*3/  2*2/                11*11*256
  conv3                     3*3/    192/ 384/  1*1/                    ReLU
  conv4                     3*3/    384/ 256/  1*1/                    ReLU
  conv5                     3*3/    256/ 256/  1*1/                    ReLU                                     3*3/  2*2/

  全连接层
  FC1



'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time


batch_size = 32
num_batches = 100

#显示网络层
def print_activation(t):
    print(t.op.name,' ',t.get_shape().as_list())


def inference(images):
    parameters = []
    #conv1
    with tf.name_scope('conv1') as scope:
        #卷积核 number=64 channels=3 size=11*11 截断的正态分布初始化卷积核参数
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        #计算卷积 strides=4*4
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        #偏差
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        #卷积结果+偏差
        bias = tf.nn.bias_add(conv, biases)
        #relu激活
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
        #lrn1数据增强
        lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn1')
        #池化核 size=3*3 strides=2*2
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print_activations(pool1)
    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)
        lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print_activations(pool2)
    #conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)
    #conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)
    #conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        print_activations(pool5)

    #FC1
    with tf.name_scope('FC1') as scope:
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight = variable_with_weight_loss(shape=[dim, 4096], stddev=0.01, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)
        FC1 = tf.nn.relu(tf.matmul(reshape, weight) + biases)
        print_activations(FC1)
    #FC2
    with tf.name_scope('FC2') as scope:
        weight = variable_with_weight_loss(shape=[4096, 4096], stddev=0.001, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)
        FC2 = tf.nn.relu(tf.matmul(FC1, weight) + biases)
        print_activations(FC2)
    #FC3
    with tf.name_scope('FC3') as scope:
        weight = variable_with_weight_loss(shape=[4096, 1000], stddev=0.001, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[1000]), dtype=tf.float32, trainable=True)
        FC3 = tf.nn.relu(tf.matmul(FC2, weight) + biases)
        print_activations(FC3)

    return FC3, parameters















