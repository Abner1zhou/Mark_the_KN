# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: 参数设定
@date:2020/2/26
"""

# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19

import argparse
from utils.path_config import data_path, data_path_q, result_path


def get_params(model):
    if model == 'cnn':
        # 获得参数
        parser = argparse.ArgumentParser(description='This is the TextCNN train project.')
        parser.add_argument('--model', default='cnn')
        parser.add_argument('-t', '--test_sample_percentage', default=0.2, type=float,
                            help='The fraction of test data.(default=0.2)')
        parser.add_argument('-p', '--padding_size', default=300, type=int,
                            help='Padding size of sentences.(default=128)')
        parser.add_argument('-e', '--embed_size', default=512, type=int, help='Word embedding size.(default=512)')
        parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
        parser.add_argument('-n', '--num_filters', default=128, type=int,
                            help='Number of each convolution kernel.(default=128)')
        parser.add_argument('-d', '--dropout_rate', default=0.1, type=float,
                            help='Dropout rate in softmax layer.(default=0.1)')
        parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=95)')
        parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float,
                            help='L2 regulation parameter.(default=0.01)')
        parser.add_argument('-b', '--batch_size', default=256, type=int, help='Mini-Batch size.(default=256)')
        parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.01)')
        parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
        parser.add_argument('--epochs', default=25, type=int, help='Number of epochs.(default=25)')
        parser.add_argument('--fraction_validation', default=0.05, type=float,
                            help='The fraction of validation.(default=0.05)')
        parser.add_argument('--results_dir', default=result_path, type=str,
                            help='The results dir including log, model, vocabulary and some images.(default=./results/)')
        parser.add_argument('--data_path', default=data_path_q, type=str,
                            help='data path')
        parser.add_argument('--vocab_save_dir', default=data_path, type=str,
                            help='vocab path')
        parser.add_argument('--workers', default=32, type=int,
                            help='use worker count.(default=32)')
    elif model == 'transformer':
        parser = argparse.ArgumentParser(description='This is the Transformer train project.')
        parser.add_argument('-p', '--padding_size', default=300, type=int,
                            help='Padding size of sentences.(default=128)')
        parser.add_argument('--data_path', default=data_path_q, type=str,
                            help='data path')
        parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
        parser.add_argument('--vocab_save_dir', default=data_path, type=str,
                            help='vocab path')
        parser.add_argument('--num_layers', default=6, type=int,
                            help='transformer encoder&decoder layers.')
        parser.add_argument('--d_model', default=256, type=int,
                            help='transformer d_model.')
        parser.add_argument('--dff', default=512, type=int,
                            help='transformer dff.')
        parser.add_argument('--num_heads', default=4, type=int,
                            help='transformer multi-heads attention.')
        parser.add_argument('-d', '--dropout_rate', default=0.1, type=float,
                            help='Dropout rate in softmax layer.(default=0.1)')
        parser.add_argument('-i', '--input_vocab_size', default=50000, type=int,
                            help='Input vocab size of Transformer.')
        parser.add_argument('-o', '--output_dim', default=95, type=int)
        parser.add_argument('-m', '--max_position_encoding', default=10000, type=int,
                            help='maximum_position_encoding')
        parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.(default=25)')
        parser.add_argument('--results_dir', default=result_path, type=str,
                            help='The results dir including log, model, vocabulary and some images.(default=./results/)')
        parser.add_argument('--buffer_size', default=50, type=int)
        parser.add_argument('-b', '--batch_size', default=64, type=int, help='Mini-Batch size.(default=256)')

    # 使用Jupyter 的时候 parser.parse_args() 会报错
    args = parser.parse_known_args()[0]
    # params = vars(args)
    return args
