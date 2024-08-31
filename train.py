# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from models.din import DeepInterestNet
from data.amazon_books.AmazonBooksDataLoad import get_amazon_books_dataloader

def train(model, train_loader, val_loader):
    # 1.设置迭代次数训练模型
    for epoch in range(epoches):
        train_loss = []
        # 1.1设置二分类交叉熵损失函数
        criterion = nn.BCELoss()
        # 1.2设置adam优化器
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        # 1.3设置模型训练，此时模型参数可以更新
        model.train()
        # 1.4遍历训练数据集，获取每个梯度的大小，输入输出
        for batch, (x, y) in enumerate(train_loader):
            # 1.4.1如果有gpu则把数据放入显存中计算，没有的话用cpu计算
            x=x.to(device)
            y=y.to(device)
            # 1.4.2数据输入模型
            pred = model(x)
            # 1.4.3计算损失
            loss = criterion(pred, y.float().detach())
            # 1.4.4优化器梯度清空
            optimizer.zero_grad()
            # 1.4.5方向传播，计算梯度
            loss.backward()
            # 1.4.6优化器迭代模型参数
            optimizer.step()
            # 1.4.7记录模型损失数据
            train_loss.append(loss.item())
        # 1.5模型固化，不修改梯度
        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
          # 1.6遍历验证数据集，获取每个梯度的大小，输入输出
          for batch, (x, y) in enumerate(val_loader):
              # 1.6.1如果有gpu则把数据放入显存中计算，没有的话用cpu计算
              x=x.to(device)
              y=y.to(device)
              # 1.6.2模型预测输入
              pred = model(x)
              # 1.6.3计算损失函数
              loss = criterion(pred, y.float().detach())
              val_loss.append(loss.item())
              prediction.extend(pred.tolist())
              y_true.extend(y.tolist())
        # 1.7计算auc得分
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        # 1.8输出模型训练效果
        print ("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (epoch, np.mean(train_loss), np.mean(val_loss), val_auc))        
    return train_loss, val_loss, val_auc

fields, train_loader, valid_Loader, test_loader = get_amazon_books_dataloader()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64,32], dropout=0.2).to(device)
# 迭代次数
epoches = 5
# 模型训练
_ = train(model)


#输入的数据
dis_test_x.apply(cate_encoder.inverse_transform).reset_index().head(1)

#模型输入的向量
test_X[0]

#预测购买的概率
model(torch.unsqueeze(test_X[0],0))

#事实上该用户是否购买
test_y[0]