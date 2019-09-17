'''
 * @desc : 实现基于LSTM的语言模型
 * @auth : TYF
 * @date : 2019/8/26 - 23:05


 RNN : 循环神经网络，时间序列的样本依次输入到一层中，时间t神经元的记忆单元会受到神经元t-1的影响
       以单词1,单词2,单词3,单词4组成的一个句子样本为例

       S(t)是t时刻的记忆单元
       S(t-1)是t-1时刻的记忆单元
       x(t)是t时刻的输入
       0(t)是t时刻的输出
       W是输入样本的权重
       U是此刻输入样本的权重
       V是输出样本的权重
       f和g均为激活函数

       S(t) = f( W*S(t-1) +U*x(t) )
       O(t) = g( V*S(t) )

       初始S(0)为0,输入单词1,单词1和S(0)加权和再激活得到S(1),S(1)激活得到O(1)
       得到S(1)后,输入单词2,单词2和S(1)加权和再激活得到S(2),S(2)激活得到O(2)
       ....
       得到S(3)后,输入单词3,单词4和S(3)加权和再激活得到S(4),S(4)激活得到O(4)
       最后输出即O(4)

       如何单词的词向量是10维,则一个句子就是(4,10)的样本,后面就需要10个神经元进行连接
       输入单词1其shape(1,10)经过十个神经元处理得到S(1)其shape(1,10)直到最后求出输出
       O(4)其shape(1,10),这样做的结果是样本(4,10)压缩成了(1,10)即4个单词得到1个语义
       而这1个语义浓缩了样本单词的构成和顺序关系

       --------------------------相关文章--------------------------------
       https://www.cnblogs.com/pinard/p/6509630.html
       --------------------------相关文章--------------------------------


 LSTM : RNN与LSTM最大的区别在于LSTM中最顶层多了一条名为"cell state"的信息传送带,其实也是信息记忆的地方
        传送带中包含几个控制门(gate)

        (1)选择忘记过去某些信息
        ft = σ( Wf[h(t-1),x(t)] + bf )  整合上一层的记忆单元h以及新的输入x 得到ft
        (2)记忆现在的某些信息
        it = σ( Wi[h(t-1),x(t)] + bi )  整合上一层的记忆单元h以及新的输入x 得到it
        _Ct = tanh( Wc[h(t-1),x(t)] + bc )  整合上一层的记忆单元h以及新的输入x 得到Ct
        (3)将过去与现在的记忆进行合并
        Ct = ft * Ct-1 + it * _Ct  整合过去的记忆Ct-1以及当前的记忆_Ct 得到当前总的记忆Ct
        (4)输出
        Ot = σ( Wo[h(t-1),x(t)] + bo )
        ht = Ot * tanh (Ct)
        可以看到最终输出O通过h得到,而h是由C得到的,和RNN区别在于信息流通道C中可以对记忆进行选择性增删


       --------------------------相关文章--------------------------------
        https://blog.csdn.net/hust_tsb/article/details/79485268
       --------------------------相关文章--------------------------------



'''
#样本下载地址 http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

from __future__ import division
import time
import numpy as np
import tensorflow as tf
import models.tutorials.rnn.ptb.reader

#定义语言模型处理输入数据的class
class PTBInput(object):
    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps #lstm的展开步数
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name = name)


#定义语言模型的class,PTBModel
class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size


#设置默认的LSTM单元
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0.0, state_is_tuple = True)
        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple = True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

#创建网络的词嵌入的部分
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype = tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

#定义输出
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:tf.get_variable_scope().reuse_variables()

                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],
                                                                    [tf.ones([batch_size * num_steps], dtype = tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

#定义学习率，优化器等
        self._lr = tf.Variable(0.0, trainable = False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape = [], name = "new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})


#利用@property装饰器可以将返回变量设为只读
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

#定义小的训练模型参数
class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

#定义中等的训练模型参数
class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


#定义大的训练模型参数
class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


#定义测试时的训练模型
class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

#定义训练一个epoch数据的函数
def run_epoch(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]

        state = vals["final_state"]

        costs += cost
        # print cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:

            print ("%.3f perplexity: %.3f speed : %.0f wps"
                %(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

#直接读取解压数据
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

#创建图
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config = config, data = train_data, name = 'TrainInput')
        with tf.variable_scope("Model", reuse = None, initializer = initializer):
            m = PTBModel(is_training = True, config = config, input_ = train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config = config, data = valid_data, name = "ValidInput")
        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mvalid = PTBModel(is_training = False, config = config, input_ = valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config = eval_config, data = test_data, name = "TestInput")

        with tf.variable_scope("Model", reuse = True, initializer = initializer):
            mtest = PTBModel(is_training = False, config = eval_config, input_ = test_input)

#创建训练的管理器
        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" %(i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op = m.train_op, verbose = True)
                print("Epoch: %d Train Perplexity: %.3f" %(i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d valid Perplexity: %.3f" %(i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" %test_perplexity)
