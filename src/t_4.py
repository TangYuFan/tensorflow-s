'''
 * @desc : tensorflow实现
 * @auth : TYF
 * @date : 2019/8/31 - 15:58
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from time import time

mnist = input_data.read_data_sets('E:/work/pycharm_space/dataSet/mnist/',one_hot=True)
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

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32,[None,784]) #输入样本x
w = tf.Variable(tf.zeros([784,10])) #权重w
b = tf.Variable(tf.zeros([10])) #偏差b
y = tf.nn.softmax(tf.matmul(x,w)+b) # y预测值
y_ = tf.placeholder(tf.float32,[None,10]) # y真实值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) #y/_y交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #优化器
tf.global_variables_initializer().run() #计算

#抓取550次每次100个样本
for i in range(int(55000/100)):
    batch_xs,batch_ys = mnist.train.next_batch(100) #样本及其标签
    train_step.run({x:batch_xs,y_:batch_ys})
    sess.run(y, feed_dict={x: batch_xs})
#只迭代一次(权重只更新一次)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #y中最大值就是预测值,y_中最大值就是真实值,这里y和_y都是onehot,这里y和_y是否相同,结果为1*55000其中0的元素是预测错误1的元素是预测准确
#正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#测试集
acc1 = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
#验证集
acc2 = sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels})

print('acc1:',acc1)
print('acc2:',acc2)


'''
#迭代10次
for j in range(30):
    #抓取550次每次100个样本
    for i in range(int(55000/100)):
        batch_xs,batch_ys = mnist.train.next_batch(100) #样本及其标签
        train_step.run({x:batch_xs,y_:batch_ys})
        sess.run(y, feed_dict={x: batch_xs})
#只迭代一次(权重只更新一次)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #y中最大值就是预测值,y_中最大值就是真实值,这里y和_y都是onehot,这里y和_y是否相同,结果为1*55000其中0的元素是预测错误1的元素是预测准确
#正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#测试集
acc1 = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
#验证集
acc2 = sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels})
print('acc1:',acc1)
print('acc2:',acc2)
'''