# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: bert data process. 因为BERT的word token直接调用源代码， 和之前的不太一样。所以需要单独构建一个数据处理文件
@date:2020/4/12
"""
import pandas as pd
import os

from model.bert.bert_model import tokenization

# Mark_the_KN/data/baidu_95.csv
DATASET = '../../data/baidu_95.csv'
DATA_OUTPUT_DIR = '../../data/bert/all_labels'
# vocab file
VOCAB_PATH = '../../data/bert/chinese_L-12_H-768_A-12/vocab.txt'

# Load data
df = pd.read_csv(DATASET, header=None, names=["labels", "item"], dtype=str)

# Create files
if not os.path.exists(os.path.join(DATA_OUTPUT_DIR, "train")):
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "train"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "valid"))
    os.makedirs(os.path.join(DATA_OUTPUT_DIR, "test"))

# Split dataset
# train:valid:test = 8:1:1
df_train = df[:int(len(df) * 0.8)]
df_valid = df[int(len(df) * 0.8):int(len(df) * 0.9)]
df_test = df[int(len(df) * 0.9):]

# Save dataset
file_set_type_list = ["train", "valid", "test"]
for file_set_type, df_data in zip(file_set_type_list, [df_train, df_valid, df_test]):
    predicate_out_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                        "predicate_out.txt"), "w",
                           encoding='utf-8')
    text_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                               "text.txt"), "w",
                  encoding='utf-8')
    token_in_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                   "token_in.txt"), "w",
                      encoding='utf-8')
    token_in_not_UNK_f = open(os.path.join(os.path.join(DATA_OUTPUT_DIR, file_set_type),
                                           "token_in_not_UNK.txt"), "w",
                              encoding='utf-8')

    """
    ##### BERT中各种Tokenizer解释#######
    BasicTokenizer的主要是进行unicode转换、标点符号分割、小写转换、中文字符分割、去除重音符号等操作， 
    最后返回的是关于词的数组（中文是字的数组）
    
    WordpieceTokenizer的目的是将合成词分解成类似词根一样的词片。
    例如将"unwanted"分解成["un", "##want", "##ed"]这么做的目的是防止因为词的过于生僻没有被收录进词典最后只能以[UNK]代替的局面，
    因为英语当中这样的合成词非常多，词典不可能全部收录。
    
    FullTokenizer的作用是对一个文本段进行以上两种解析，
    最后返回词（字）的数组，同时还提供token到id的索引以及id到token的索引。
    这里的token可以理解为文本段处理过后的最小单元。
    """
    # Processing
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_PATH, do_lower_case=True)  # 初始化 bert_token 工具
    # feature
    text = '\n'.join(df_data.item)
    text_tokened = df_data.item.apply(bert_tokenizer.tokenize)
    text_tokened = '\n'.join([' '.join(row) for row in text_tokened])
    text_tokened_not_UNK = df_data.item.apply(bert_tokenizer.tokenize_not_UNK)
    text_tokened_not_UNK = '\n'.join([' '.join(row) for row in text_tokened_not_UNK])
    # label only choose first 3 lables: 高中 学科 一级知识点
    # if you want all labels
    # just remove list slice
    predicate_list = df_data.labels.apply(lambda x: x.split())
    predicate_list_str = '\n'.join([' '.join(row) for row in predicate_list])

    print(f'datasize: {len(df_data)}')
    text_f.write(text)
    token_in_f.write(text_tokened)
    token_in_not_UNK_f.write(text_tokened_not_UNK)
    predicate_out_f.write(predicate_list_str)

    text_f.close()
    token_in_f.close()
    token_in_not_UNK_f.close()
    predicate_out_f.close()




