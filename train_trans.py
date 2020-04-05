# -*- coding: UTF-8 -*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com
@desc: transformer train file
@date:2020/3/29
"""
from model.transformer.model import Transformer
from model.transformer.layers import create_padding_mask, get_optimizer, get_loss
from utils.params_utils import get_params
from utils.metrics import micro_f1,macro_f1
from utils.data_proc import data_loader

import time
import tensorflow as tf


def config_gpu(use_cpu=False):
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # gpu报错 使用cpu运行
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3600)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                print(e)


def main():
    config_gpu()
    params = get_params("transformer")
    X_train, X_test, y_train, y_test, vocab, mlb = data_loader(params, is_rebuild_dataset=False)
    y_train = tf.constant(y_train, tf.float32)
    y_test = tf.constant(y_test, tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(params.buffer_size,
                                          reshuffle_each_iteration=True).batch(params.buffer_size, drop_remainder=True)

    test_dataset=test_dataset.batch(params.batch_size)
    # 流水线技术 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，
    # CPU 开始准备步骤 N + 1 的数据。这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。
    # 没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态:
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    transformer = Transformer(params.num_layers, params.d_model, params.num_heads, params.dff,
                              params.input_vocab_size, params.output_dim,
                              params.max_position_encoding,
                              rate=params.dropout_rate)

    optimizer = get_optimizer()
    train_loss, train_accuracy = get_loss()
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, params.results_dir, max_to_keep=3)
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ]

    @tf.function()
    def train_step(inp, tar):
        enc_padding_mask = create_padding_mask(inp)

        with tf.GradientTape() as tape:
            predictions = transformer(inp, True, enc_padding_mask=enc_padding_mask)
            loss = loss_function(tar, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar, predictions)

        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    def predict(inp, tar, enc_padding_mask):
        predictions = transformer(inp, False, enc_padding_mask=enc_padding_mask)
        mi_f1 = micro_f1(tar, predictions)
        ma_f1 = macro_f1(tar, predictions)
        return mi_f1, ma_f1

    for epoch in range(params.epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            mic_f1, mac_f1 = train_step(inp, tar)

            if batch % 50 == 0:
                test_input, test_target = next(iter(test_dataset))
                enc_padding_mask = create_padding_mask(test_input)
                val_mic_f1, val_mac_f1 = predict(test_input, test_target, enc_padding_mask)

                print(
                    'Epoch {} Batch {} Loss {:.4f} micro_f1 {:.4f} macro_f1 {:.4f} val_micro_f1 {:.4f} val_macro_f1 {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), mic_f1, mac_f1, val_mic_f1, val_mac_f1))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == '__main__':
    main()
