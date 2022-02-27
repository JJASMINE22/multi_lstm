# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：net.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Lambda
from tensorflow.keras.models import Model, Sequential
from CustomLayers import MultiHeadsAttention

class Mha_Lstm_cell(Layer):
    """
    于LSTM单元间嵌入多头注意力机制,
    屏蔽后序列对前序列的作用
    """
    def __init__(self,
                 embedding_size: int,
                 multihead_num: int,
                 target_channels: int,
                 dropout: float,
                 **kwargs):
        super(Mha_Lstm_cell, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.target_chs = target_channels
        self.dropout = dropout

        self.lstm_in = LSTM(units=self.embedding_size,
                            return_sequences=True)
        self.mha = MultiHeadsAttention(embedding_size=self.embedding_size,
                                       multihead_num=self.multihead_num,
                                       dropout=self.dropout)
        self.lstm_out = LSTM(units=self.target_chs,
                             return_sequences=False)

    def get_config(self):
        config = super(Mha_Lstm_cell, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'multihead_num': self.multihead_num,
            'target_channels': self.target_chs,
            'dropout': self.dropout
        })

    def sequence_mask(self, x):

        mask = 1 - tf.linalg.band_part(tf.ones(shape=(tf.shape(x)[1], )*2), -1, 0)
        mask = mask[tf.newaxis, ...]

        return mask

    def call(self, input, training=None, mask=None):

        x = self.lstm_in(input)

        seq_mask = self.sequence_mask(x)

        x, att = self.mha([x, x, x], mask=seq_mask)

        output = self.lstm_out(x)

        return output

class Lstm_cell(Layer):
    def __init__(self,
                 embedding_size: int,
                 target_channels: int,
                 **kwargs):
        super(Lstm_cell, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.target_chs = target_channels

        self.lstm_in = LSTM(units=self.embedding_size,
                            return_sequences=True)
        self.lstm_out = LSTM(units=self.target_chs,
                             return_sequences=False)

    def get_config(self):
        config = super(Lstm_cell, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'target_channels': self.target_chs
        })

    def call(self, input, training=None, mask=None):

        x = self.lstm_in(input)
        output = self.lstm_out(x)

        return output

class CreateModel(Model):
    """
    包含3个mha单元与6个lstm单元
    """
    def __init__(self,
                 tgt_channels: int,
                 embed_size: int,
                 multihead_num: int,
                 mha_lstm_cell_nums: int,
                 lstm_cell_nums: int,
                 dropout: float,
                 **kwargs):
        super(CreateModel, self).__init__(**kwargs)
        self.lstm_cell_nums = lstm_cell_nums
        self.mha_lstm_cell_nums = mha_lstm_cell_nums

        self.mha_lstm_cells = [Mha_Lstm_cell(embedding_size=embed_size,
                                             multihead_num=multihead_num,
                                             target_channels=tgt_channels,
                                             dropout=dropout) for _ in range(mha_lstm_cell_nums)]

        self.lstm_cells = [Lstm_cell(embedding_size=embed_size,
                                     target_channels=tgt_channels) for _ in range(lstm_cell_nums)]

    def call(self, inputs, training=None, mask=None):

        outputs = []
        for i in range(self.mha_lstm_cell_nums):
            outputs.append(self.mha_lstm_cells[i](inputs[i]))

        for i in range(self.mha_lstm_cell_nums, self.mha_lstm_cell_nums+self.lstm_cell_nums):
            outputs.append(self.lstm_cells[i-self.mha_lstm_cell_nums](inputs[i]))

        return outputs
