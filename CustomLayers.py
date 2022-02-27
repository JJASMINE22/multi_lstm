# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：CustomLayers.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2


class LayerNormalization(Layer):
    """
    层归一化, 主要应用于自然语言处理
    """
    def __init__(self,
                 epsilon=1e-6,
                 center=True,
                 scale=True,
                 gamma_initializer=initializers.Ones(),
                 beta_initializer=initializers.Zeros(),
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer
        self.gamma_regularizer = gamma_regularizer
        self.beta_regularizer = beta_regularizer
        self.gamma_constraint = gamma_constraint
        self.beta_constraint = beta_constraint

    def build(self, input_shape):

        super(LayerNormalization, self).build(input_shape)

        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=True if self.scale or self.center else False,
                                     name='gamma')

        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint,
                                    trainable=True if self.scale or self.center else False,
                                    name='beta')

        self.built

    def call(self, inputs, *args, **kwargs):
        # 针对通道维执行标准化
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        var = tf.reduce_sum(tf.math.pow((inputs - mean), 2), axis=-1, keepdims=True) / inputs.shape[-1]
        normalize = tf.math.divide((inputs - mean), tf.sqrt(var + self.epsilon))

        if tf.logical_or(self.center, self.scale):
            normalize = tf.nn.bias_add(tf.multiply(self.gamma, normalize), self.beta)

        return normalize

class MultiHeadsAttention(Layer):
    """
    多头注意力
    """
    def __init__(self,
                 embedding_size=None,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=l2(5e-4),
                 bias_regularizer=l2(5e-4),
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_attention_bias=False,
                 use_attention_activation=False,
                 multihead_num=None,
                 dropout=None,
                 **kwargs):
        super(MultiHeadsAttention, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_attention_bias = use_attention_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.use_attention_activation = use_attention_activation
        self.multihead_num = multihead_num
        self.dropout = dropout
        self.layernorm = LayerNormalization()

    def get_config(self):

        config = super(MultiHeadsAttention, self).get_config()
        config.update({
            'embedding_size': self.embedding_size,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'use_attention_bias': self.use_attention_bias,
            'bias_initializer': self.bias_initializer,
            'bias_regularizer': self.bias_regularizer,
            'bias_constraint': self.bias_constraint,
            'use_attention_activation': self.use_attention_activation,
            'multihead_num': self.multihead_num,
            'dropout': self.dropout
        })

        return config

    def build(self, input_shape):

        self.kernel_q = self.add_weight(shape=(input_shape[0][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_q')

        self.kernel_k = self.add_weight(shape=(input_shape[1][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_k')

        self.kernel_v = self.add_weight(shape=(input_shape[-1][-1], self.embedding_size),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        name='kernel_v')

        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.embedding_size),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      name='kernel')

        if self.use_attention_bias:
            self.attention_bias = self.add_weight(shape=(1,),
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint,
                                                  trainable=True,
                                                  name='attention_bias')

        self.built = True

    def call(self, inputs, mask=None, *args, **kwargs):
        """
        编码器中q、k、v均为src输入源特征
        解码器中q为tgt目标特征, k与v均为src输入源特征
        """
        assert 2 <= len(inputs) < 4

        q = tf.matmul(inputs[0], self.kernel_q)
        k = tf.matmul(inputs[1], self.kernel_k)
        v = tf.matmul(inputs[-1], self.kernel_v) if len(inputs) == 3 \
            else tf.matmul(inputs[1], self.kernel_v)
        seq_len_q, seq_len_k, seq_len_v = q.shape[1], k.shape[1], v.shape[1]

        # q = tf.reshape(q, shape=[-1, seq_len_q, self.embedding_size//self.multihead_num])
        # k = tf.reshape(k, shape=[-1, seq_len_k, self.embedding_size//self.multihead_num])
        # v = tf.reshape(v, shape=[-1, seq_len_v, self.embedding_size//self.multihead_num])
        # 区别与原文的reshape, 此处用split→concat实现多头, 并还原
        q = tf.concat(tf.split(q, num_or_size_splits=self.multihead_num, axis=-1), axis=0)
        k = tf.concat(tf.split(k, num_or_size_splits=self.multihead_num, axis=-1), axis=0)
        v = tf.concat(tf.split(v, num_or_size_splits=self.multihead_num, axis=-1), axis=0)

        attention = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.embedding_size // self.multihead_num,
                                                                        dtype=tf.float32))
        if self.use_attention_bias:
            attention += self.attention_bias

        if self.use_attention_activation:
            attention = tf.nn.tanh(attention)

        if mask is not None:
            # 仅使用sequence_mask, 不复制维度, 直接通过broadcast运算
            attention -= 1e+9 * mask

        attention = tf.nn.softmax(attention)

        context = tf.matmul(attention, v)

        context = tf.concat(tf.split(context, num_or_size_splits=self.multihead_num, axis=0), axis=-1)
        # context = tf.reshape(context, shape=[-1, seq_len_q, self.embedding_size])
        output = tf.matmul(context, self.kernel)

        output = tf.nn.dropout(output, rate=self.dropout)

        output = tf.add(inputs[0], output)

        output = self.layernorm(output)

        return output, attention
