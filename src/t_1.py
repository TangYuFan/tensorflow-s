'''
 * @desc : 张量运算仿真神经网络的运行
 * @auth : TYF
 * @date : 2019/8/31 - 15:01
'''

import tensorflow as tf
import numpy as np

#隐层创建
#input_dim:输入神经元个数
#output_dim:输出神经元个数
#inputs:输入样本
def layer_debug(input_dim,output_dim,inputs,activation=None):
      W = tf.Variable(tf.random_normal([input_dim,output_dim])) # W(input_dim,output_dim)  random_normal随机生成正太分布的随机数矩阵
      B = tf.Variable(tf.random_normal([1,output_dim])) # B(1,output_dim)  random_normal随机生成正太分布的随机数矩阵
      XWB = tf.matmul(inputs,W)+B;  #计算 x * w + b
      if activation==None: # 如果不需要激活
          outputs = XWB
      else:#如果需要激活
          outputs = activation(XWB)
      #当层输出、权重、偏差
      return outputs,W,B



#建立计算图
X = tf.placeholder("float",[None,4])   #输入样本X(1,4)
h1,w1,b1 = layer_debug(input_dim=4,output_dim=3,inputs=X,activation=tf.nn.relu)   #创建第一隐层 返回第一隐层的  输出  样本与第一隐层之间的W 样本与第一隐层之间的B
h2,w2,b2 = layer_debug(input_dim=3,output_dim=2,inputs=h1,activation=tf.nn.relu)   #创建第二隐层 返回第二隐层的  输出  第一隐层与第二隐层之间的W 第一隐层与第二隐层之间的B


#运行计算图
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4,0.2,0.4,0.5]]) #1行4列样本
    #获取输出
    (_h1,_h2,_w1,_w2,_b1,_b2) = sess.run((h1,h2,w1,w2,b1,b2),feed_dict={X:X_array})
    print('----------------')
    print('_X(样本):',X_array)
    print('_w1(样本与第一隐层的W):',_w1)
    print('_b1(样本与第一隐层的B):',_b1)
    print('_h1(第一隐层输出):',_h1)
    print('_w2(第一隐层与第二隐层的W):',_w2)
    print('_b2(第一隐层与第二隐层的B):',_b2)
    print('_h2(最终输出):',_h2)
