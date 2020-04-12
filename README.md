# Mark_the_KN

> 数据来源：百度智慧课堂  <https://study.baidu.com/tiku>

数据集说明：

高中历史、地理、政治、生物

一共26243条数据，标签类别：95

## 主要目标：

1. 对高中试题进行分类
2. 对试题进行知识点标注（多标签分类）

## 使用的方法

1. Bayes
2. FastText
3. TextCNN
4. GCN
5. Transformer
6. Bert
7. ERINE

## TextCNN

**主要参数：**

- filter sizes = 3, 4, 5
- filter numbers = 128
- epochs = 25
- batch sizes = 256



**结果：**

Micro f1: 0.84510887

Macro f1: 0.6661619



## Transformer

**主要参数：**

- padding-size  = 300
- layer numbers = 6
- d_model = 256
- head numbers = 4
- batch size = 64

**结果：**

- Epoch 10 Loss 0.0366 Accuracy 0.9894
- Micro f1: 0.8451088666915894
- Macro f1: 0.6661618947982788

## BERT

目录说明：

bert_model : BERT官方代码