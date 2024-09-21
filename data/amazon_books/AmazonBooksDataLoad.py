
import pandas as pd
import torch.utils.data
import os
from utils.utils import LoadPick, DumpPick

POSTFIX = ['uid', 'material', 'cat', 'len']
"""# 三.数据处理
"""

def GetIdsMap(data, train_path='data/amazon_books/amazon-books-100k.txt', load_seq_len=50):
    cache_path = '/'.join(train_path.split('/')[0:-1] + ["cache/"])
    file_dict = cache_path + 'dic_cache'
    if os.path.exists(file_dict):
        print("load amazon data maps from pickle")
        return (LoadPick(cache_path + postfix) for postfix in POSTFIX) 
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
    maps = [userMap, materialMap, cateMap, lenMap]
    for idx, dic in enumerate(maps): 
        DumpPick(dic, cache_path + POSTFIX[idx]) 
    with open(file_dict, 'w') as f: pass
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
        hist_item = torch.tensor(([self.data_df.iloc[idx][col] for col in self.hist_feats[0]]))
        hist_cate = torch.tensor(([self.data_df.iloc[idx][col] for col in self.hist_feats[1]]))
        label =torch.tensor(self.data_df.iloc[idx]['label'])
        #x = self.data_df.iloc[idx][:-1]
        #y = self.data_df.label.values
        x = (user, item_feat, hist_item, hist_cate)
        y = label
        return x, y

def get_amazon_books_dataloader(train_path="amazon-books-100k.txt", batch_size=4096, hist_len=10):
    print("Start loading amazon books data....")
    prefix = "data/amazon_books/"
    train_path = prefix + train_path
    print(train_path)
    data = pd.read_csv(train_path)
    userMap, materialMap, cateMap, lenMap= GetIdsMap(data)
    data, user_feat, hist_feats, item_feat= AmazonBookPreprocess(data, userMap, materialMap, cateMap, lenMap, hist_len)
    field_dims = len(userMap) + len(materialMap) + len(cateMap) + len(lenMap)
    #模型输入
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
    #feat_size = (user_feat.size(), hist_feats.size(), item_feat.size())
    feat_num = (len(feat_cols) for feat_cols in [user_feat, hist_feats, item_feat])
    return feat_num, field_dims, train_loader, valid_Loader, test_loader

if __name__ == '__main__':
    prefix = "./"
    train_path="tmp"
    train_path = prefix + train_path
    print(train_path)
    data = pd.read_csv(train_path)
    userMap, materialMap, cateMap, lenMap = GetIdsMap(data)
    #data, user_feat, hist_feat, item_feat = AmazonBookPreprocess(data, userMap, materialMap, cateMap, lenMap, 3)
    #print(data)
    #get_amazon_books_dataloader(train_path="tmp")
