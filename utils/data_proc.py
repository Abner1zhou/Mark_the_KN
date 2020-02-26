# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: 数据预处理, 数据加载
@date:2020/2/25
"""

import re
import os
import jieba
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from utils.path_config import sw_path, root
from utils.params_utils import get_params
from utils.multi_cpus import parallelize


def get_stop_words(stop_words_path=sw_path):
    """
    处理停用词表
    :param stop_words_path: 停用词表路径
    :return: 停用词表list
    """
    stop_words = []
    with open(stop_words_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    return stop_words


def clean_sentence(sentence):
    """
    特殊符号去除
    使用正则表达式去除无用的符号、词语
    """
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！？、~@#￥%……&*（）]+|题目',
            '', sentence)
    else:
        return ''


def cut_words(sentences):
    """
    :param sentences: 待处理句子
    :param swp: stop words path
    :return: 切词 去除停用词以后的List
    """
    # 清除无用词
    sentence = clean_sentence(sentences)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    stop_words = get_stop_words()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


# 用于并行处理数据，传入到parallelize函数里
def parallelize_df(df):
    df['content'] = df['content'].apply(cut_words)
    return df


def proc(df, params):
    """
    :param df: 待处理数据  （完整的句子）
    :param params: 词表大小
    :return: 切词，去除停用词以后的数据
    """
    # 数据并行预处理，清洗
    df = parallelize(df, parallelize_df)
    # word2index
    text_preprocessor = Tokenizer(num_words=params.vocab_size, oov_token="<UNK>")
    text_preprocessor.fit_on_texts(df['content'])
    x = text_preprocessor.texts_to_sequences(df['content'])
    word_dict = text_preprocessor.word_index
    return word_dict, x


def data_loader(params, is_rebuild_dataset=False):
    if os.path.exists(os.path.join(root, 'data', 'X_train.npy')) and not is_rebuild_dataset:
        x_train = np.load(os.path.join(root, 'data', 'X_train.npy'))
        x_test = np.load(os.path.join(root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(root, 'data', 'y_test.npy'))
        return x_train, x_test, y_train, y_test

    # 读取数据
    df = pd.read_csv(params.data_path, header=None).rename(columns={0: 'label', 1: 'content'})
    word_dict, x = proc(df, params)
    # save vocab
    with open(params.vocab_save_dir + '/vocab.txt', 'w', encoding='utf-8') as f:
        for k, v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')
    # padding
    x = pad_sequences(x, maxlen=params.padding_size, padding='post', truncating='post')
    # 划分标签
    df['label'] = df['label'].apply(lambda x: x.split())
    # 多标签编码
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['label'])
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # 保存数据
    np.save(os.path.join(root, 'data', 'X_train.npy'), x_train)
    np.save(os.path.join(root, 'data', 'X_test.npy'), x_test)
    np.save(os.path.join(root, 'data', 'y_train.npy'), y_train)
    np.save(os.path.join(root, 'data', 'y_test.npy'), y_test)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    params = get_params()
    print('Parameters:', type(params))
    X_train, X_test, y_train, y_test = data_loader(params)
    print(X_train)
