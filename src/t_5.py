'''
 * @desc : tensorflow实现 自编码
 * @auth : TYF
 * @date : 2019/9/5 - 18:56

稀疏编码(Sparse Coding)算法是一种无监督学习方法，
它用来寻找一组“超完备”基向量来更高效地表示样本数据。
稀疏编码算法的目的就是找到一组基向量 ，使得我们能将输入向量 表示为这些基向量的线性组合。

'''

#图像的组合
#一个图片可以提取出很多基本  特征,而这些基本特征的线性组合可以组合成原图像的碎片(高阶特征),进而组合成原图像(非像素点复制而是原图像的另一种表达)
#在识别任务中,图像经过网络后得到很多高阶特征,使用这些高阶特征就能实现识别任务
#这个称为矩阵的稀疏表达

#自编码器
#1.输入和输出一致
#2.希望使用高阶特征来重构自己

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Xavier初始化器
#因为初始化权重太小则会在传递过程中逐渐缩小而难以产生作用
#初始化太大会在传递过程中放大导致发散和失效
#作用:让权重满足0均值,同时方差为2/(in+out)
#此处创建一个正负constant * np.sqrt(6.0/(fan_in+fan_out)范围内的均匀分布
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform(
        (fan_in,fan_out),
        minval=low,
        maxval=high,
        dtype=tf.float32
    )


#去噪自编码器
class AdditiveGaussianNoiseAutoencoder(object):
    # n_input 输入变量number
    # n_hidden 隐层节点数
    # transfer_function 隐层激活函数(默认softplus)
    # optimizer 优化器(默认Adam)
    # scale 高斯噪声(默认0.1)
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdadeltaOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        #定义网络结构
        self.x=tf.placeholder(tf.float32,[None,self.n_input]) #输入n_input维向量
        self.hidden = self.transfer(
            tf.add(
                tf.matmul(
                    self.x + scale * tf.random_normal((n_input,)),
                    self.weights['w1']
                ),
                self.weights['b1']
            )
        )
        self.reconstruction = tf.add(
                tf.matmul(
                    self.hidden,
                    self.weights['w2']
                ),
                self.weights['b2']
        )
        #定义损失函数
        self.cost= 0.5 * tf.reduce_sum(
                tf.pow(
                    tf.subtract(
                        self.reconstruction,
                        self.x
                    ),
                    2.0
                )
        )
        self.optimizer = optimizer.minimize(self.cost)
        #Sess
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    #权重初始化
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    #计算一个batch数据的cost和optimizer
    def partial_fit(self,X):
        cost,opt = self.sess.run(
            (self.cost,self.optimizer),
            feed_dict={self.x:X,self.scale:self.training_scale}
        )
        return cost

    #求损失
    def calc_total_cost(self,X):
        return self.sess.run(
            self.cost,
            feed_dict={self.x:X,self.scale:self.training_scale}
        )

    #获取隐层输出结果
    def transform(self,X):
        return self.sess.run(
            self.hidden,
            feed_dict={self.x:X,self.scale:self.training_scale}
        )

    #将隐层输出重建为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(
            self.reconstruction,
            feed_dict={self.hidden:hidden}
        )

    #重建函数
    def reconstruction(self,X):
        return self.sess.run(
            self.reconstruction,
            feed_dict={self.x:X,self.scale:self.training_scale}
        )

    #权重/偏差get函数
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


#数据标准化
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

#随机抓取batch_size个样本
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

#Load数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#标准化
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)
#样本总个数
n_samples = int(mnist.train.num_examples)
#迭代次数
training_epochs = 20
#抓取size
batch_size = 128
#每1次迭代统计一次损失cost
display_step = 1
#自编码器
autoencoder = AdditiveGaussianNoiseAutoencoder(
    n_input=784,
    n_hidden=200,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.001),
    scale=0.01
)
#迭代
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size) #每次抓取128个样本,抓取55000/128次
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size) #抓取样本
        cost = autoencoder.partial_fit(batch_xs) #计算损失
        avg_cost += cost / n_samples * batch_size
    #是否显示损失
    if epoch % display_step ==0:
        print('epoch:','%04d'%(epoch+1),'cost:','{:.9f}'.format(avg_cost))
#测试集计算总损失
print('total cost:',str(autoencoder.calc_total_cost(X_test)))