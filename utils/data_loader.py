# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: 数据预处理
@date:2020/2/25
"""

import re
import jieba
from keras_preprocessing.text import Tokenizer
import sys
sys.path.append('../')
from utils.path_config import sw_path


def get_stop_words(stop_words_path):
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


def cut_words(sentences, swp=sw_path):
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
    # TODO 停用词路径
    stop_words = get_stop_words(swp)
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


def proc(df, params):
    df['content'] = df['content'].apply(cut_words)

    text_preprocessor = Tokenizer(num_words=params.vocab_size, oov_token="<UNK>")
    text_preprocessor.fit_on_texts(df['content'])
    x = text_preprocessor.texts_to_sequences(df['content'])
    word_dict = text_preprocessor.word_index
    return df
