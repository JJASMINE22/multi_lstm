# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import config as cfg
import tensorflow as tf
from multi_unit_lstm import Multi_Unit_Lstm
from utils.data_generator import Generator


if __name__ == '__main__':

    gen = Generator(txt_path=cfg.text_path,
                    batch_size=cfg.batch_size,
                    ratio=cfg.train_ratio,
                    time_seq=cfg.time_seq,
                    imf_nums=cfg.imf_nums,
                    decompose=cfg.decompose)

    multi_lstm = Multi_Unit_Lstm(target_channels=cfg.tgt_channels,
                                 embedding_size=cfg.embed_size,
                                 multihead_num=cfg.multihead_num,
                                 mha_lstm_cell_nums=cfg.mha_lstm_cell_nums,
                                 lstm_cell_nums=cfg.lstm_cell_nums,
                                 dropout=cfg.dropout,
                                 learning_rate=cfg.learning_rate)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(multi_lstm=multi_lstm.model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_func = gen.imfs_generate(training=True)
    val_func = gen.imfs_generate(training=False)
    for epoch in range(cfg.epoches):

        for i in range(gen.get_train_len()):
            multi_lstm.train(*next(train_func))

        for i in range(gen.get_val_len()):
            multi_lstm.validate(*next(val_func))

        print(
            f'Epoch {epoch + 1}, '
            f'train_loss:  {multi_lstm.train_metrics_loss.result()}, '
            f'test_loss: {multi_lstm.val_metrics_loss.result()}'
        )

        ckpt_save_path = ckpt_manager.save()

        multi_lstm.train_metrics_loss.reset_states()
        multi_lstm.val_metrics_loss.reset_states()
