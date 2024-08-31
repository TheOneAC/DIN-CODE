# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""# 四.模型定义

#### 1.这块代码主要是，定义din模型，模型的内容可以看https://arxiv.org/pdf/1706.06978.pdf 了解din模型是个啥
#### 2.pytorch定义的模型主要是看init和forward，
#### 3.init的功能是初始化一些变量，和别的类定义一样；
#### 4.forward的功能是向前传播，是调用类时直接将参数输入forward中，进行计算；
"""

class Dice(nn.Module):
    """
    自定义的dice激活函数，原论文有公式介绍，有点复杂我也没看懂，别的地方用的不多，不介绍了。
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9
    
    def forward(self, x):

        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1-p) + x.mul(p)
    
        return x

class ActivationUnit(nn.Module):
    """
    激活函数单元
    功能是计算用户购买行为与推荐目标之间的注意力系数，比如说用户虽然用户买了这个东西，但是这个东西实际上和推荐目标之间没啥关系，也不重要，所以要乘以一个小权重
    """
    def __init__(self, embedding_dim, dropout=0.2, fc_dims = [32, 16]):
        super(ActivationUnit, self).__init__()
        # 1.初始化fc层
        fc_layers = []
        # 2.输入特征维度
        input_dim = embedding_dim*4     
        # 3.fc层内容：全连接层（4*embedding,32）—>激活函数->dropout->全连接层（32,16）->.....->全连接层（16,1）
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        # 4.将上面定义的fc层，整合到sequential中
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, query, user_behavior):
        """
            :param query:targe目标的embedding ->（输入维度） batch*1*embed 
            :param user_behavior:行为特征矩阵 ->（输入维度） batch*seq_len*embed
            :return out:预测目标与历史行为之间的注意力系数
        """
        # 1.获取用户历史行为序列长度
        seq_len = user_behavior.shape[1]
        # 2.序列长度*embedding
        queries = torch.cat([query] * seq_len, dim=1)
        # 3.前面的把四个embedding合并成一个（4*embedding）的向量，
        #  第一个向量是目标商品的向量，第二个向量是用户行为的向量，
        #  至于第三个和第四个则是他们的相减和相乘（这里猜测是为了添加一点非线性数据用于全连接层，充分训练）
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior, 
                                queries * user_behavior], dim = -1)
        out = self.fc(attn_input)
        return out

class AttentionPoolingLayer(nn.Module):
    """
      注意力序列层
      功能是计算用户行为与预测目标之间的系数，并将所有的向量进行相加，这里的目的是计算出用户的兴趣的能力向量
    """
    def __init__(self, embedding_dim,  dropout):
        super(AttentionPoolingLayer, self).__init__()
        self.active_unit = ActivationUnit(embedding_dim = embedding_dim, 
                                          dropout = dropout)
        
    def forward(self, query_ad, user_behavior, mask):
        """
          :param query_ad:targe目标x的embedding   -> （输入维度） batch*1*embed
          :param user_behavior:行为特征矩阵     -> （输入维度） batch*seq_len*embed
          :param mask:被padding为0的行为置为false  -> （输入维度） batch*seq_len*1
          :return output:用户行为向量之和，反应用户的爱好
        """
        # 1.计算目标和历史行为之间的相关性
        attns = self.active_unit(query_ad, user_behavior)     
        # 2.注意力系数乘以行为 
        output = user_behavior.mul(attns.mul(mask))
        # 3.历史行为向量相加
        output = user_behavior.sum(dim=1)
        return output

class DeepInterestNet(nn.Module):
    """
      模型主体
      功能是用户最近的历史40个购买物品是xxx时，购买y的概率是多少
    """

    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super(DeepInterestNet, self).__init__()
        # 1.特征维度，就是输入的特征有多少个类
        self.feature_dim = feature_dim
        # 2.embeding层，将特征数值转化为向量
        self.embedding = nn.Embedding(feature_dim+1, embed_dim)
        # 3.注意力计算层（论文核心）
        self.AttentionActivate = AttentionPoolingLayer(embed_dim, dropout)
        # 4.定义fc层
        fc_layers = []
        # 5.该层的输入为历史行为的embedding，和目标的embedding，所以输入维度为2*embedding_dim
        #  全连接层（2*embedding,fc_dims[0]）—>激活函数->dropout->全连接层（fc_dims[0],fc_dims[1]）->.....->全连接层（fc_dims[n],1）
        input_dim = embed_dim * 2      
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        fc_layers.append(nn.Linear(input_dim, 1))
        # 6.将所有层封装
        self.mlp = nn.Sequential(*fc_layers)        
    
    def forward(self, x):
        """
            x输入(behaviors*40,ads*1) ->（输入维度） batch*(behaviors+ads)
            
        """
        # 1.排除掉推荐目标
        behaviors_x = x[:,:-1]
        # 2.记录之前填充为0的行为位置
        mask = (behaviors_x > 0).float().unsqueeze(-1)
        # 3.获取推荐的目标
        ads_x = x[:,-1]
        # 4.对推荐目标进行向量嵌入
        query_ad = self.embedding(ads_x).unsqueeze(1)
        # 5.对用户行为进行embeding，注意这里的维度为(batch*历史行为长度*embedding长度)
        user_behavior = self.embedding(behaviors_x)
        # 6.矩阵相乘，将那些行为为空的地方全部写为0
        user_behavior = user_behavior.mul(mask)
        # 7.将用户行为乘上注意力系数,再把所有行为记录向量相加
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)
        # 8.将计算后的用户行为行为记录和推荐的目标进行拼接
        concat_input = torch.cat([user_interest, query_ad.squeeze(1)], dim = 1)
        # 9.输入用户行为和目标向量，计算预测得分
        out = self.mlp(concat_input)
        # 10.sigmoid激活函数
        out = torch.sigmoid(out.squeeze(1))        
        return out

"""*斜体文本*# 六.模型训练

#### 模型训练的一般步骤：
#### 1.定义损失函数
#### 2.定义优化器
#### 3.定义模型参数可更新
#### 4.遍历数据集训练模型
#####  $\qquad$*4.1输入数据，获得预测结果
#####  $\qquad$*4.2计算损失
#####  $\qquad$*4.3反向传播
#####  $\qquad$*4.4参数更新
"""

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

# 定义din模型
model = DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64,32], dropout=0.2).to(device)
# 迭代次数
epoches = 5
# 模型训练
_ = train(model)





"""# 7.效果展示

#### 最后拿一条数据看下效果
"""

#输入的数据
dis_test_x.apply(cate_encoder.inverse_transform).reset_index().head(1)

#模型输入的向量
test_X[0]

#预测购买的概率
model(torch.unsqueeze(test_X[0],0))

#事实上该用户是否购买
test_y[0]