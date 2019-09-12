'''
 * @desc : tensorflow实现 word2Vec
 * @auth : TYF
 * @date : 2019/8/31 - 15:58

  word2Vec :  一般的字可以根据字典库n转为一个n维的onehot编码，
              但是维数太高且稀疏，所以需要转为维数少的稠密向量
              训练一个空间，输入单次的onehot编码得到其空间稠密向量，
              且语义相近的单次其稠密向量在空间位置更加接近。
              称为词嵌入。

  预测模型 :  CBOW主要用来从原始语句推测目标词汇，skip-gram用来从目标词汇推测原始语境

  这里实现skip-gram

'''

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

#样本下载
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename
filename = maybe_download('text8.zip', 31344016)
#解压成单词list
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words = read_data(filename)
print('Data size', len(words))


print('--------------------------------------------------------')

#字典库大小50000
vocabulary_size = 50000

#创建vocabulary词汇表
def build_dataset(words):
    count = [['UNK', -1]]
    #Counter统计集合中单词频数返回一个 word:num 字典
    #most_common取top n
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    #遍历word:num字典一共5000
    for word, _ in count:
        dictionary[word] = len(dictionary)
    #dictionary字典 按照频数高到低顺序排序 序列就是该单词的编码
    #{'UNK': 0, 'the': 1, 'of': 2, 'and': 3, 'one': 4, 'in': 5, 'a': 6, ',.....}
    data = list()
    unk_count = 0
    #遍历样本单词列表
    for word in words:
        if word in dictionary:
            index = dictionary[word]  #如果字库中存在该单词,则返回该单词的编号(编码)
        else:  #如果不存在则编号为0
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # data 编码
    # count 每个单词频数
    # dictionary 词汇表
    # reverse_dictionary 词汇表反转形式
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words #删除单词list节省内存

print('---------------------------------------------------------------------')
print('Most common words (+UNK)',count[:5]) #取频数排名前5
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]]) #前10个单词及其编号
print('---------------------------------------------------------------------')


'''
skip-Gram模式是目标单词推语境 :
原始数据
    the quick brown for jumped over the lazy dog

batch_size : 一个batch的大小
skip_windows : 指单词最远可以联系的距离
num_skips : 对每个单词生成多少个样本

以目标单词左右相邻一个空隙(skip_windows=1)的语料,一个语料能生成2个训练样本(num_skips=2)为例 :

语境  目标  语境       生成的样本
the   quick brown      (quick,the) (quick,brown)
quick brown for        (brown,quick) (brown.for)

'''
data_index = 0

#生成训练用的batch数据
def generatt_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0  #batch_size必须是num_skips的整数倍(确保每个batch包含了一个词汇对应的所有样本)
    assert num_skips <= 2 * skip_window  #num_skips不能大于skip_windows的两倍
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 用np.ndarray将batch和labels初始化为数组。
    span = 2 * skip_window + 1  # span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
    buffer = collections.deque(maxlen=span)  # 创建一个最大容量为span的deque(双向队列）
    for _ in range(span):#从data_index开始，把span个单词顺序读入buffer作为初始值
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    #第一层循环，每次循环内对一个目标单词生成样本。
    for i in range(batch_size // num_skips):
        target = skip_window  #buffer中第skip_window个变量为目标单词。
        targets_to_avoid = [ skip_window ] #targets_to_avoid为生成样本时需要避免的单词列表
        #第二层循环，每次循环对一个语境单词生成样本，先产生随机数，，
        for j in range(num_skips):
            while target in targets_to_avoid:  #直到随机数不在targets_to_avoid中代表可以使用的语境单词，然后产生一个样本
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


#调用generatt_batch函数测试一下其功能
batch, labels = generatt_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
          reverse_dictionary[labels[i, 0]])

print('--------------------------------------')

batch_size = 128  #一个batch有128个样本
embedding_size = 128   #每个样本的每个单词转为128维的稠密向量
skip_window = 1  #每个样本的每个单词转为128维的稠密向量
num_skips = 2 #一个目标单词生成两个训练样本

valid_size = 16  # 用来抽取的验证单词数
valid_window = 100  # 验证单词只从频数最高的100个单词中抽取，
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # valid_examples为验证数据集
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量

graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  #将前面随即产生的valid_examples转化为tensorflow中的constant

    with tf.device('/cpu:0'):
        #生成正态分布-1.0到1.0之间的随机矩阵 5000*128
        #相当生成了字典库中所有单词的词向量
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        #查找训练样本在字典库中的词向量并返回
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        #nec loss作为训练的优化目标
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    #使用tf.nn.nce_loss计算学习出现的词向量embedding在训练数据上的loss，并在 tf.reduve_mean进行汇总。
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size)
    )
    #随机梯度下降 学习速率1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #计算词向量的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    #得到词向量标准化后的词向量
    normalized_embeddings = embeddings / norm
    #查询该标准化后的词向量在验证集中的嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    #计算验证单词的嵌入向量与词汇表中所有单词的相似性
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    #初始化所有模型参数
    init = tf.global_variables_initializer()

#最大迭代次数为10万次
num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    average_loss = 0
    for step in range(num_steps):
        #抓取一个batch的数据
        batch_inputs, batch_labels = generatt_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        #计算
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        #每2000次循环，计算一下平均loss并打印
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ":", average_loss)
            average_loss = 0
        #每10000次循环，计算一次验证单词与全部单词的相似度，打印最相似的8个单词
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = "Nearest to %s: " % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
            final_embeddings = normalized_embeddings.eval()



#模型可视化
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than enbeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    plt.show()
#原始的128维嵌入向量降到2维方便画图观察
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100  # 这里只显示词频最高的100个单词的可视化结果
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)