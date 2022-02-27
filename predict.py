# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：predict.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
import config as cfg
import matplotlib.pyplot as plt
from model.net import CreateModel
from utils.data_generator import Generator

if __name__ == '__main__':

    gen = Generator(txt_path=cfg.text_path,
                    batch_size=cfg.batch_size,
                    ratio=cfg.train_ratio,
                    time_seq=cfg.time_seq,
                    imf_nums=cfg.imf_nums,
                    decompose=cfg.decompose)

    model = CreateModel(tgt_channels=cfg.tgt_channels,
                        embed_size=cfg.embed_size,
                        multihead_num=cfg.multihead_num,
                        mha_lstm_cell_nums=cfg.mha_lstm_cell_nums,
                        lstm_cell_nums=cfg.lstm_cell_nums,
                        dropout=cfg.dropout)

    ckpt = tf.train.Checkpoint(multi_lstm=model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    total_predictions = model(gen.val_source)
    total_targets = gen.val_target
    for i in range(cfg.tgt_channels):
        for j in range(cfg.imf_nums):
            plt.subplot(cfg.imf_nums, 1, j+1)
            plt.plot(total_predictions[j][:, i], color='r', linewidth=0.3, label='feature_{:0>1d}_prediction'.format(i))
            plt.plot(total_targets[j][:, i], color='y', linewidth=0.3, label='feature_{:0>1d}_real'.format(i))
            plt.grid(True)
            if not j:
                plt.legend(loc='upper right', fontsize='x-small')
        plt.show()

