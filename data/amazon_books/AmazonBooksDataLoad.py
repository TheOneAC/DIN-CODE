
import shutil
import struct
from collections import defaultdict
from pathlib import Path
import pandas as pd

##import lmdb
import numpy as np
import torch.utils.data
import torch.utils.data as Data
##from tqdm import tqdm
from sklearn.model_selection import train_test_split


"""# 三.数据处理

#### 1.由于该数据集只有10w条，很多商品id只出现了一次，故编码的时候是以类别作为编码和预测的targe
#### 2.如果你要用学生做的试题序列作为训练集，且这些试题被不同的学生来回做过，可以用试题作为唯一的编码
#### 3.这里数据处理的目的是形成学生答题序列，把文本数据转化为唯一的数值编码，作为模型的输入，用于预测的目标试题
"""

def AmazonBookPreprocess(dataframe, seq_len=40):
    """
    数据集处理
    :param dataframe: 未处理的数据集
    :param seq_len: 数据序列长度
    :return data: 处理好的数据集
    """
    # 1.按'|'切割，用户历史购买数据，获取item的序列和类别的序列
    data = dataframe.copy()
    data['hist_item_list'] = dataframe.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
    data['hist_cate_list'] = dataframe.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

    # 2.获取cate的所有种类，为每个类别设置一个唯一的编码
    cate_list = list(data['cateID'])
    _ = [cate_list.extend(i) for i in data['hist_cate_list'].values]
    # 3.将编码去重
    cate_set = set(cate_list + ['0'])  # 用 '0' 作为padding的类别

    # 4.截取用户行为的长度,也就是截取hist_cate_list的长度，生成对应的列名
    cols = ['hist_cate_{}'.format(i) for i in range(seq_len)]

    # 5.截取前40个历史行为，如果历史行为不足40个则填充0
    def trim_cate_list(x):
        if len(x) > seq_len:
            # 5.1历史行为大于40, 截取后40个行为
            return pd.Series(x[-seq_len:], index=cols)
        else:
            # 5.2历史行为不足40, padding到40个行为
            pad_len = seq_len - len(x)
            x = x + ['0'] * pad_len
            return pd.Series(x, index=cols)

    # 6.预测目标为试题的类别
    labels = data['label']
    data = data['hist_cate_list'].apply(trim_cate_list).join(data['cateID'])

    # 7.生成类别对应序号的编码器，如book->1,Russian->2这样
    cate_encoder = LabelEncoder().fit(list(cate_set))
    # 8.这里分为两步，第一步为把类别转化为数值，第二部为拼接上label
    data = data.apply(cate_encoder.transform).join(labels)
    return data

def get_amazon_books_dataloader(train_path="amazon-books-100k.txt", batch_size=4096):
    print("Start loading amazon books data....")
    prefix = "../amazon_books/"
    train_path = prefix + train_path
    print(train_path)
    data = pd.read_csv(train_path)
    data = AmazonBookPreprocess(data)

    #模型输入
    data_X = data.iloc[:,:-1]
    #模型输出
    data_y = data.label.values
    #划分训练集，测试集，验证集
    tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.1, random_state=42, stratify=data_y)
    train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.1, random_state=42, stratify=tmp_y)
    # numpy转化为torch
    train_X = torch.from_numpy(train_X.values).long()
    val_X = torch.from_numpy(val_X.values).long()
    test_X = torch.from_numpy(test_X.values).long()

    train_y = torch.from_numpy(train_y).long()
    val_y = torch.from_numpy(val_y).long()
    test_y = torch.from_numpy(test_y).long()
    # 设置dataset
    train_set = Data.TensorDataset(train_X, train_y)
    val_set = Data.TensorDataset(val_X, val_y)
    test_set = Data.TensorDataset(test_X, test_y)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_Loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    # field_dims = dataset.field_dims
    field_dims = data.max().max()
    # print(field_dims)
    # print(sum(field_dims))
    return field_dims, train_loader, valid_Loader, test_loader
