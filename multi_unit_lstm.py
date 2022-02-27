# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：multi_unit_lstm.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''

import tensorflow as tf
from model.net import CreateModel

class Multi_Unit_Lstm:
    def __init__(self,
                 target_channels: int,
                 embedding_size: int,
                 multihead_num: int,
                 mha_lstm_cell_nums: int,
                 lstm_cell_nums: int,
                 dropout: float,
                 learning_rate: float,
                 **kwargs):

        self.model = CreateModel(tgt_channels=target_channels,
                                 embed_size=embedding_size,
                                 multihead_num=multihead_num,
                                 mha_lstm_cell_nums=mha_lstm_cell_nums,
                                 lstm_cell_nums=lstm_cell_nums,
                                 dropout=dropout)

        self.imf_nums = mha_lstm_cell_nums + lstm_cell_nums

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.train_loss, self.val_loss = 0, 0

        self.train_metrics_loss = tf.keras.metrics.Mean()
        self.val_metrics_loss = tf.keras.metrics.Mean()

    @tf.function
    def train(self, sources, targets):
        '''
        :param sources: 9个imf分量的批量时序数据
        :param targets: 9个imf分量的批量预测目标
        '''
        sources = [tf.squeeze(_, axis=1) for _ in tf.split(sources, num_or_size_splits=self.imf_nums, axis=1)]
        targets = [tf.squeeze(_, axis=1) for _ in tf.split(targets, num_or_size_splits=self.imf_nums, axis=1)]
        with tf.GradientTape() as tape:
            predictions = self.model(sources)
            for target, prediction in zip(targets, predictions):
                self.train_loss += self.loss(target, prediction)
            self.train_loss/=self.imf_nums
        gradients = tape.gradient(self.train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics_loss(self.train_loss)
        self.train_loss = 0

    def validate(self, sources, targets):
        sources = [tf.squeeze(_, axis=1) for _ in tf.split(sources, num_or_size_splits=self.imf_nums, axis=1)]
        targets = [tf.squeeze(_, axis=1) for _ in tf.split(targets, num_or_size_splits=self.imf_nums, axis=1)]
        predictions = self.model(sources)
        for target, prediction in zip(targets, predictions):
            self.val_loss += self.loss(target, prediction)
        self.val_loss /= self.imf_nums

        self.val_metrics_loss(self.val_loss)
        self.val_loss = 0
