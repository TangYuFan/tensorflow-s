'''
 * @desc : mlp训练mnist
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


def plot_image(image,label):
    plt.imshow(image.reshape(28,28),cmap='binary') #样本
    plt.title('class='+label) #标签
    plt.show()

#显示样本矩阵
plot_image(mnist.train.images[5],str(np.argmax(mnist.train.labels[5])))  #标签是onehot编码转成数字

#层定义
def layer(in_dim,out_dim,input,activation=None):
    W = tf.Variable(tf.random_normal([in_dim,out_dim]))  #权重
    B = tf.Variable(tf.random_normal([1,out_dim]))  #偏差
    XWB = tf.matmul(input,W)+B
    if activation==None:
        output = XWB
    else:
        output = activation(XWB)
    return output


#正向传播
x = tf.placeholder('float',[None,784]) #样本1*784
h1 = layer(in_dim=784,out_dim=1000,input=x,activation=tf.nn.relu)  #输入层 第一隐层
h2 = layer(in_dim=1000,out_dim=1000,input=h1,activation=tf.nn.relu)  #第二隐层
h3 = layer(in_dim=1000,out_dim=10,input=h2,activation=None)   #输出层
#反向传播
y = tf.placeholder('float',[None,10])
#损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3,labels=y)) #1.函数softmax_cross_entropy_with_logits交叉熵  2.logits设置真实值  3.labels设置预测值 4.reduce_mean是取平均值
#优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function) #设置学习率
#评分器
is_predict_right = tf.equal(tf.argmax(y,1),tf.argmax(h3,1)) #判断真实值y和预测值h3是否相等  正确返回1错误返回0
accuracy = tf.reduce_mean(tf.cast(is_predict_right,'float')) #计算1和0的比例
#迭代次数
epoch = 15
#抓取size
batch_size = 100
#抓取批次
batch_time = int(mnist.train.num_examples/batch_size) #55000(样本总数)/100(每批次样本抓取数)=550(批次数)


#记录每次迭代的误差、准确率用于最后作图
lost_list = []
epoch_list = []
accuracy_list = []

#计算开始时间
start_time = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#迭代epoch(15)次
for e in range(epoch):
    #每次迭代分550批次抓取
    for i in range(batch_time):
         #每批次抓取 得到100个样本及其标签
         batch_x,batch_y = mnist.train.next_batch(batch_size)
         #该批次样本输入到网络,指定优化器
         sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})  #x,y是placeholder定义的占位符,这里将每批次样本及其标签推给两个占位符
    #完成本次迭代,用验证集数据计算误差、准确率
    loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y:mnist.validation.labels}) #x,y是placeholder定义的占位符,这里将验证集的样本和标签推给两个占位符
    #记录本次迭代评分
    epoch_list.append(e)
    accuracy_list.append(acc)
    lost_list.append(loss)
    #打印本次迭代评分
    print('epoch=','%2d'%(e+1),'accuracy=',acc,'lost=','{:.9f}'.format(loss))
#计算训练总耗时
duration = time()-start_time
print('duration=',duration)


#作图
#epoch= 15 accuracy= [0.8538, 0.8974, 0.9136, 0.9206, 0.9258, 0.9322, 0.9336, 0.9366, 0.9414, 0.9396, 0.9404, 0.9432, 0.9448, 0.9426, 0.9458] lost= 1.313437343
#误差率
fig =  plt.gcf()
fig.set_size_inches(4,2) #图大小4行4列
plt.plot(epoch_list,lost_list,label ='loss') #xy轴数据
plt.ylabel('loss') #y轴label
plt.xlabel('epoch') #x轴label
plt.legend(['loss'],loc='upper left')
#准确率
plt.plot(epoch_list,accuracy_list,label ='accuracy')
fig =  plt.gcf()
fig.set_size_inches(4,2) #图大小4行4列
plt.ylim(0.8,1)
plt.ylabel('accuracy') #y轴label
plt.xlabel('epoch') #x轴label
plt.legend()
plt.show()


#计算测试机准确率和误差
print('test accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
#获取预测结果
prediction_result = sess.run(tf.argmax(h3,1),feed_dict={x:mnist.test.images}) #h2输出层的输出是onehot编码 使用argmax还原
#对比预测值和输出值:
for i in range(len(prediction_result)):
    print('样本',i,'predict=',prediction_result[i],'label=',mnist.test.labels[i])




