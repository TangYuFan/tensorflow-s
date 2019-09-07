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

  卷积层       输入          卷积核(Ksize/channels/number/stride)       激活      输出          数据增强          池化核(Psize/stride)       输出
  conv1       224*224*3     11*11/  3/   64/   4*4/                    ReLU     55*55*64       LRN              3*3/  2*2/                27*27*64
  conv2       27*27*64      5*5/    64/  192/  1*1/                    ReLU     27*27*192      LRN              3*3/  2*2/                13*13*192
  conv3       13*13*192     3*3/    192/ 384/  1*1/                    ReLU     13*13*384
  conv4       13*13*384     3*3/    384/ 256/  1*1/                    ReLU     13*13*256
  conv5       13*13*256     3*3/    256/ 256/  1*1/                    ReLU     13*13*256                       3*3/  2*2/                6*6*256

  全连接层    输入节点   激活      输出节点
  FC1         9216      ReLU      4096
  FC2         4096      ReLU      4096
  FC3         9216      Softmax   1000



'''
import tensorflow as tf



batch_size = 32
num_batches = 100


#初始化权重
def variable_with_weight_loss(shape,stddev,wl):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

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
        parameters += [kernel, biases]
        #lrn1数据增强
        lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn1')
        #池化核 size=3*3 strides=2*2
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print('------')
        print('conv1 input:',images.shape)
        print('conv1 output:',conv1.get_shape().as_list())
        print('conv1 pool output:',pool1.get_shape().as_list())
    #conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        print('------')
        print('conv2 input:',pool1.get_shape().as_list())
        print('conv2 output:',conv2.get_shape().as_list())
        print('conv2 pool output:',pool2.get_shape().as_list())
    #conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print('------')
        print('conv3 input:',pool2.get_shape().as_list())
        print('conv3 output:',conv3.get_shape().as_list())
    #conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print('------')
        print('conv4 input:',conv3.get_shape().as_list())
        print('conv4 output:',conv4.get_shape().as_list())
    #conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        print('------')
        print('conv5 input:',conv4.get_shape().as_list())
        print('conv5 output:',conv5.get_shape().as_list())
        print('conv5 pool output:',pool5.get_shape().as_list())

    #FC1
    with tf.name_scope('FC1') as scope:
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight = variable_with_weight_loss(shape=[dim, 4096], stddev=0.01, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)
        FC1_d = tf.nn.dropout(tf.matmul(reshape, weight) + biases,0.25)
        FC1 = tf.nn.relu(FC1_d)
        print('------')
        print('FC1 input:',reshape.get_shape().as_list())
        print('FC1 output:',FC1.get_shape().as_list())
    #FC2
    with tf.name_scope('FC2') as scope:
        weight = variable_with_weight_loss(shape=[4096, 4096], stddev=0.001, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)
        FC2_d  = tf.nn.dropout(tf.matmul(FC1, weight) + biases,0.25);
        FC2 = tf.nn.relu(FC2_d)
        print('------')
        print('FC2 input:',FC1.get_shape().as_list())
        print('FC2 output:',FC2.get_shape().as_list())
    #FC3
    with tf.name_scope('FC3') as scope:
        weight = variable_with_weight_loss(shape=[4096, 1000], stddev=0.001, wl=0.004)
        biases = tf.Variable(tf.constant(0.0, shape=[1000]), dtype=tf.float32, trainable=True)
        FC3 = tf.nn.softmax(tf.matmul(FC2, weight) + biases)
        print('------')
        print('FC3 input:', FC2.get_shape().as_list())
        print('FC3 output:', FC3.get_shape().as_list())
    return FC3, parameters




if __name__ == '__main__':

    #创建输入图像的一个batch
    image_size = 224
    images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
    FC3,parameters = inference(images)
    #初始化网络中的所有变量
    init = tf.global_variables_initializer()
    #执行网络
    sess = tf.Session()
    sess.run(init)













