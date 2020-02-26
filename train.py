# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: TextCNN训练
@date:2020/2/26
"""
import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from pprint import pprint

from utils.data_proc import data_loader
from model.text_cnn import text_cnn
from utils.metrics import micro_f1, macro_f1
from utils.params_utils import get_params


def train(x_train, x_test, y_train, y_test, params, save_path):
    print("\nTrain...")
    model = build_model(params)

    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    # parallel_model.compile(tf.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy',
    #                        metrics=[micro_f1, macro_f1])
    # keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(args.results_dir, timestamp, "model.pdf"))
    # y_train = tf.one_hot(y_train, args.num_classes)
    # tb_callback = keras.callbacks.TensorBoard(os.path.join(args.results_dir, timestamp, 'log/'),
    #                                           histogram_freq=0.1, write_graph=True,
    #                                           write_grads=True, write_images=True,
    #                                           embeddings_freq=0.5, update_freq='batch')

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')

    history = model.fit(x_train, y_train,
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        workers=params.workers,
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(x_test, y_test))

    print("\nSaving model...")
    keras.models.save_model(model, save_path)
    pprint(history.history)


def build_model(params):
    if params.model == 'cnn':
        model = text_cnn(max_sequence_length=params.padding_size, max_token_num=params.vocab_size,
                         embedding_dim=params.embed_size,
                         output_dim=params.num_classes)
        model.compile(tf.optimizers.Adam(learning_rate=params.learning_rate),
                      loss='binary_crossentropy',
                      metrics=[micro_f1, macro_f1])

    else:

        pass

    model.summary()
    return model


if __name__ == '__main__':
    params = get_params()
    print("Params:", params)
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(params.results_dir, timestamp))
    os.mkdir(os.path.join(params.results_dir, timestamp, 'log/'))

    X_train, X_test, y_train, y_test = data_loader(params)

    train(X_train, X_test, y_train, y_test, params, os.path.join(params.results_dir, 'TextCNN.h5'))