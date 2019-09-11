
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
import  math
import os
import  random
import zipfile
import numpy as np
import  urllib
import tensorflow as tf


url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception('GG!')
    return filename

filename = maybe_download('text8.zip',31344016)






