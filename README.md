## Multiple LSTM模型的tensorflow2实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤) 
6. [预测效果 Predict](#预测效果)

## 所需环境
1. Python3.7
2. tensorflow-gpu>=2.0  
3. Numpy==1.19.5
4. CUDA 11.0+
5. Pandas==1.2.4
6. Scipy==1.6.3
7. Pyhht==0.1.0
8. Matplotlib==3.2.2

## 模型结构
由9个单元组成，其中多头注意力LSTM单元3个，LSTM单元6个，两种单元结构如下：

LSTM单元
![image]()

MHA LSTM单元
![image]()

## 注意事项
1. 模型由9个单元组成，共两种类型：LSTM单元与多头注意力LSTM单元
2. 根据经验模态分解，将输入数据拆分为9组，并按pearson相关系数排序
3.  模型误差由9组imf分量共同决定，提高泛化性
4.  数据路径、训练参数均位于config.py

## 文件下载    
链接：https://pan.baidu.com/s/1PYSmTSc7ucK3nYCpyEeb5A 
提取码：sets
下载解压后放置于config.py中设置的路径即可。
## 训练步骤
1.默认设置某气候数据为训练样本
2.运行train.py

## 预测效果
9分量的特征1：
![image]()
9分量的特征2：
![image]()

