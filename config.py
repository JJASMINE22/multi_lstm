# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：config.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
# ===multi unit lstm neural network===

# data_generator
text_path = 'C:\\DATASET\\time_series_data\\weather_data.xlsx' # '数据文件绝对路径'
time_seq = 7
train_ratio = 0.7
imf_nums = 9
decompose = True

# model
tgt_channels = 10 # depends on the setting of the data prediction dimensions
embed_size = 256
multihead_num = 4
# mha_lstm_cell_nums add lstm_cell_nums must equal to imf nums
mha_lstm_cell_nums = 3
lstm_cell_nums = 6
dropout = 0.3

# training
epoches = 150
batch_size = 4
learning_rate = 1e-3
ckpt_path = '.\\tf_models\\checkpoint'
