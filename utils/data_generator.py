# -*- coding: UTF-8 -*-
'''
@Project ：multi_lstm
@File    ：data_generator.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import math
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from pyhht import EMD
from scipy.stats import pearsonr


class Generator(object):
    """
    包含标准数据生成模式与经验模态分解数据生成模式
    """
    def __init__(self,
                 txt_path: str,
                 batch_size: int,
                 ratio: float,
                 time_seq: int,
                 imf_nums: int,
                 decompose: bool):
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.ratio = ratio
        self.time_seq = time_seq
        self.imf_nums = imf_nums
        self.decompose = decompose
        self.radian = np.pi / 180
        self.train_source, self.train_target, self.val_source, self.val_target = self.split_train_val()

    def filtering_imfs(self, x):
        """
        执行emd操作并筛选
        :param x:输入特征向量
        :return: 相关系数帅选下的imf分量
        """
        emd = EMD(x)
        imfs = emd.decompose()
        corrs = [pearsonr(x, imf)[0] for imf in imfs]
        index = np.arange(len(corrs))
        # np.argsort(corrs)[-self.imf_nums:]
        sorted_index = sorted(index, key=lambda i: corrs[i], reverse=True)
        filtered_imfs = imfs[sorted_index][:self.imf_nums]
        return filtered_imfs

    def split_train_val(self):
        """
        此方法实现数据的归一化处理, 通过属性decompose判断数据是否进行经验模态分解
        :return: 训练时序数据、训练目标、验证时序数据、验证目标
        """
        if self.decompose:
            total_data = self.preprocess()
            imfs = [self.filtering_imfs(total_data[:, i])
                    for i in range(total_data.shape[-1])]
            imfs = np.stack(imfs, axis=-1)
            imfs = [np.squeeze(imf, axis=0) for imf in np.split(imfs, indices_or_sections=imfs.shape[0], axis=0)]

            imfs = [2*(imf-imf.min(axis=0))/(imf.max(axis=0)-imf.min(axis=0))-1 for imf in imfs]
            seq_sources = [np.array([np.array(imf)[i:i+self.time_seq]
                                     for i in range(len(imf)-self.time_seq)])
                           for imf in imfs]
            targets = [np.array([np.array(imf)[i+self.time_seq]
                                 for i in range(len(imf)-self.time_seq)])
                       for imf in imfs]
            index = np.arange(len(seq_sources[0]))
            np.random.shuffle(index)
            train_source = [seq_source[index[:int(len(index) * self.ratio)]] for seq_source in seq_sources]
            train_target = [target[index[:int(len(index) * self.ratio)]] for target in targets]
            val_source = [seq_source[index[int(len(index) * self.ratio):]] for seq_source in seq_sources]
            val_target = [target[index[int(len(index) * self.ratio):]] for target in targets]

            return train_source, train_target, val_source, val_target

        total_data, seq_source, target = self.preprocess()
        index = np.arange(len(seq_source))
        np.random.shuffle(index)
        train_source = seq_source[index[:int(len(index)*self.ratio)]]
        train_target = target[index[:int(len(index)*self.ratio)]]
        val_source = seq_source[index[int(len(index)*self.ratio):]]
        val_target = target[index[int(len(index)*self.ratio):]]

        return train_source, train_target, val_source, val_target

    def preprocess(self):
        '''
        原始数据中可能存在空字符
        找到所有空字符的索引，并为其赋值
        '''
        df = pd.read_excel(self.txt_path, keep_default_na=False)
        df['时间'] = df['时间'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
        df = df.set_index('时间')
        df[df == ''] = np.nan

        index = pd.date_range(df.index[0], df.index[-1], freq='D')
        df = df.reindex(index, fill_value=np.nan)
        df = df.drop(columns=['城市', '天气', '风向', '风级(级)', '日降雨量(mm)'])

        df = df.astype(float)
        df = df.interpolate()

        x_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.cos(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(len(df)))))

        y_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.sin(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(len(df)))))

        df.insert(loc=4, column='横向风速', value=x_speed)
        df.insert(loc=5, column='纵向风速', value=y_speed)
        df = df.drop(columns=['风速(m/s)'])
        # -1~1 normalize

        if self.decompose:
            return np.array(df)

        df = 2 * (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0)) - 1
        assign_source = np.array([np.array(df)[i:i + self.time_seq] for i in range(len(df) - self.time_seq)])
        assign_target = np.array([np.array(df)[i + self.time_seq] for i in range(len(df) - self.time_seq)])

        return np.array(df), assign_source, assign_target

    def get_train_len(self):
        train_len = len(self.train_source)
        if self.decompose:
            train_len = len(self.train_source[0])
        if not train_len % self.batch_size:
            return train_len//self.batch_size
        else:
            return train_len//self.batch_size + 1

    def get_val_len(self):
        val_len = len(self.val_source)
        if self.decompose:
            val_len = len(self.val_source[0])

        if not val_len % self.batch_size:
            return val_len//self.batch_size
        else:
            return val_len//self.batch_size + 1

    def generate(self, training=True):
        """
        标准数据生成
        """
        while True:
            if training:
                idx = np.arange(len(self.train_source))
                np.random.shuffle(idx)
                train_source = self.train_source[idx]
                train_target = self.train_target[idx]
                targets, sources = [], []
                for i, (src, tgt) in enumerate(zip(train_source, train_target)):
                    sources.append(src)
                    targets.append(tgt)
                    if np.equal(len(targets), self.batch_size) or np.equal(i+1, train_source.shape[0]):
                        annotation_targets, annotation_sources = targets.copy(), sources.copy()
                        targets.clear()
                        sources.clear()
                        yield np.array(annotation_sources), np.array(annotation_targets)

            else:
                idx = np.arange(len(self.val_source))
                np.random.shuffle(idx)
                val_source = self.val_source[idx]
                val_target = self.val_target[idx]
                targets, sources = [], []
                for i, (src, tgt) in enumerate(zip(val_source, val_target)):
                    sources.append(src)
                    targets.append(tgt)
                    if np.equal(len(targets), self.batch_size) or np.equal(i+1, val_source.shape[0]):
                        annotation_targets, annotation_sources = targets.copy(), sources.copy()
                        targets.clear()
                        sources.clear()
                        yield np.array(annotation_sources), np.array(annotation_targets)

    def imfs_generate(self, training=True):
        """
        emd数据生成
        """
        while True:
            if training:
                idx = np.arange(len(self.train_source[0]))
                np.random.shuffle(idx)
                sources = [source[idx] for source in self.train_source]
                targets = [target[idx] for target in self.train_target]
                imf_targets = []
                imf_sources = []
                for i in range(sources[0].shape[0]):
                    imf_sources.append(np.array([sources[j][i] for j in range(self.imf_nums)]))
                    imf_targets.append(np.array([targets[j][i] for j in range(self.imf_nums)]))

                    if np.equal(len(imf_sources), self.batch_size) or np.equal(i+1, sources[0].shape[0]):
                        annotation_targets, annotation_sources = np.array(imf_targets).copy(), np.array(imf_sources).copy()
                        imf_targets.clear()
                        imf_sources.clear()
                        yield annotation_sources, annotation_targets

            else:
                idx = np.arange(len(self.val_source[0]))
                np.random.shuffle(idx)
                sources = [source[idx] for source in self.val_source]
                targets = [target[idx] for target in self.val_target]
                imf_targets = []
                imf_sources = []
                for i in range(sources[0].shape[0]):
                    imf_sources.append(np.array([sources[j][i] for j in range(self.imf_nums)]))
                    imf_targets.append(np.array([targets[j][i] for j in range(self.imf_nums)]))

                    if np.equal(len(imf_sources), self.batch_size) or np.equal(i + 1, sources[0].shape[0]):
                        annotation_targets, annotation_sources = np.array(imf_targets).copy(), np.array(
                            imf_sources).copy()
                        imf_targets.clear()
                        imf_sources.clear()
                        yield annotation_sources, annotation_targets
