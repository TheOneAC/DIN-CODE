
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
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
import os
import tqdm

"""# 三.数据处理

#### 1.由于该数据集只有10w条，很多商品id只出现了一次，故编码的时候是以类别作为编码和预测的targe
#### 2.如果你要用学生做的试题序列作为训练集，且这些试题被不同的学生来回做过，可以用试题作为唯一的编码
#### 3.这里数据处理的目的是形成学生答题序列，把文本数据转化为唯一的数值编码，作为模型的输入，用于预测的目标试题
"""
def GetIdsMap(data, load_seq_len=50):
    userID_set =set(list(data['userID']))
    itemID_set =set(list(data['itemID']))
    cateID_set =set(list(data['cateID']))  
    for item in data['hist_cate_list'].values:
        for id in item.strip('\'').split('|'):
            cateID_set.add(id)
    for item in data['hist_item_list'].values:
        for id in item.strip('\'').split('|'):
            itemID_set.add(id)
    map_offset = 1
    userMap = {id:idx+map_offset for idx, id in enumerate(userID_set)}
    map_offset += len(userID_set)
    materialMap = {id:idx+map_offset for idx, id in enumerate(itemID_set)}
    map_offset += len(itemID_set)
    cateMap = {id:idx+map_offset for idx, id in enumerate(cateID_set)}
    map_offset += len(cateID_set)
    lenMap = {idx:idx+map_offset for idx in range(load_seq_len)}
    userMap['0']=0
    materialMap['0']=0
    cateMap['0']=0
    lenMap['0']=0
    #print(cateMap)
    return userMap, materialMap, cateMap, lenMap

def AmazonBookPreprocess(dataframe, userMap, materialMap, cateMap, lenMap, hist_len):
    """
    数据集处理
    :param dataframe: 未处理的数据集
    :param seq_len: 数据序列长度
    :return data: 处理好的数据集
    """
    data = pd.DataFrame()
    #data['hist_len'] = dataframe['hist_item_list'].apply(lambda x: min(len(x.split('|')), hist_len)).map(lenMap)
    def trim_seq_list(x, id_map, cols, seq_len=hist_len):
        x = x.split('|')
        if len(x) > seq_len:
            seq = pd.Series(x[-seq_len:], index=cols)
        else:
            pad_len = seq_len - len(x)
            x = x + ['0'] * pad_len
            seq = pd.Series(x, index=cols)
        return seq.map(id_map)

    cate_cols = ['hist_cate_{}'.format(i) for i in range(hist_len)]
    data_hist_cate = dataframe['hist_cate_list'].apply(lambda x: trim_seq_list(x, cateMap, cate_cols))
    for col_name in cate_cols:
        data[col_name] = data_hist_cate[col_name] 
    item_cols = ['hist_item_{}'.format(i) for i in range(hist_len)]
    data_hist_item = dataframe['hist_item_list'].apply(lambda x: trim_seq_list(x, materialMap, item_cols))
    for col_name in item_cols:
        data[col_name] = data_hist_item[col_name] 
    #data_hist_item = dataframe['hist_item_list'].apply(lambda x: trim_seq_list(x, materialMap, prefix='his_item'))
    data['userID'] = dataframe['userID'].map(userMap)
    data['itemID'] = dataframe['itemID'].map(materialMap)
    data['cateID'] = dataframe['cateID'].map(cateMap)
    data['label'] = dataframe['label']
    user_feat = ['userID']
    item_feat = [ 'itemID', 'cateID']
    hist_feats = [item_cols, cate_cols]

    return data, user_feat, hist_feats, item_feat

class AmazonBooksData(torch.utils.data.Dataset):
    def __init__(self, data, user_feat, hist_feats, item_feat):
        self.data_df = data
        self.user_feat = user_feat
        self.hist_feats = hist_feats
        self.item_feat = item_feat

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        user = torch.tensor([self.data_df.iloc[idx][col] for col in self.user_feat])
        item_feat = torch.tensor([self.data_df.iloc[idx][col] for col in self.item_feat]) 
        hist_item = torch.tensor([self.data_df.iloc[idx][col] for col in self.hist_feats[0]])
        hist_cate = torch.tensor([self.data_df.iloc[idx][col] for col in self.hist_feats[1]])
        #x = self.data_df.iloc[idx][:-1]
        #y = self.data_df.label.values
        x = (user, item_feat, hist_item, hist_cate)
        y = self.data_dfp['label']
        return x, y

def get_amazon_books_dataloader(train_path="amazon-books-100k.txt", batch_size=4096, hist_len=40):
    print("Start loading amazon books data....")
    prefix = "data/amazon_books/"
    train_path = prefix + train_path
    print(train_path)
    data = pd.read_csv(train_path)
    userMap, materialMap, cateMap, lenMap= GetIdsMap(data)
    data, user_feat, hist_feats, item_feat = AmazonBookPreprocess(data, userMap, materialMap, cateMap, lenMap, 3)
    field_dims = len(userMap) + len(materialMap) + len(cateMap) + len(lenMap)
    #模型输入
    '''
    
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
    '''
    all_dataset = AmazonBooksData(data, user_feat, hist_feats, item_feat)
    train_size = int(0.9 * len(all_dataset))
    test_size = len(all_dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size - test_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_Loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    #field_dims = all_dataset.field_dims
    #fields = data.max().max()
    # print(field_dims)
    # print(sum(field_dims))
    return field_dims, train_loader, valid_Loader, test_loader

if __name__ == '__main__':
    prefix = "./"
    train_path="tmp"
    train_path = prefix + train_path
    print(train_path)
    data = pd.read_csv(train_path)
    userMap, materialMap, cateMap, lenMap = GetIdsMap(data)
    data, user_feat, hist_feat, item_feat = AmazonBookPreprocess(data, userMap, materialMap, cateMap, lenMap, 3)
    print(data)