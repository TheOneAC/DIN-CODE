# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.layers import MLP, ActivationUnit

class DinAttentionLayer(nn.Module):
    """
      注意力序列层
      功能是计算用户行为与预测目标之间的系数，并将所有的向量进行相加，这里的目的是计算出用户的兴趣的能力向量
    """
    def __init__(self, hist_embedding_dim,  dropout, hist_len):
        super(DinAttentionLayer, self).__init__()
        self.active_unit = ActivationUnit(hist_embedding_dim = hist_embedding_dim, 
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

class DIN(nn.Module):
    """
      模型主体
      功能是用户最近的历史40个购买物品是xxx时，购买y的概率是多少
    """

    def __init__(self, feature_dim, embed_dim, feat_num, mlp_layers, dropout):
        super(DIN, self).__init__()
        # 1.特征维度，就是输入的特征有多少个类
        self.feature_dim = feature_dim
        # 2.embeding层，将特征数值转化为向量
        self.embedding = nn.Embedding(feature_dim+1, embed_dim)
        # 3.注意力计算层（论文核心）
        user_feat_num, hist_feat_num,  item_feat_num = feat_num
        self.item_att_dim = embed_dim*hist_feat_num
        self.AttentionActivate = DinAttentionLayer(self.item_att_dim, dropout)
        # 5.该层的输入为历史行为的embedding，和目标的embedding，所以输入维度为2*embedding_dim
        #  全连接层（2*embedding,fc_dims[0]）—>激活函数->dropout->全连接层（fc_dims[0],fc_dims[1]）->.....->全连接层（fc_dims[n],1）
        self.input_dim = (user_feat_num + item_feat_num)* embed_dim + self.item_att_dim
        self.mlp = MLP(mlp_layers, input_dim=self.input_dim, dropout = dropout)
    
    def forward(self, x):
        """
            x输入(behaviors*40,ads*1) ->（输入维度） batch*(behaviors+ads)
            
        """
        # 1.排除掉推荐目标
        user, item_feat, hist_item, hist_cate = x
        #print(hist_item)
        #print(hist_item.dim())
        # 2.记录之前填充为0的行为位置
        mask = (hist_item > 0).float().unsqueeze(-1)
        user_embeddings = self.embedding(user).squeeze() 
        # 4.对推荐目标进行向量嵌入
        item_embedding = self.embedding(item_feat)
        item_embeddings= torch.flatten(item_embedding, start_dim=1)
        
        # 5.对用户行为进行embeding，注意这里的维度为(batch*历史行为长度*embedding长度)
        user_behavior_item = self.embedding(hist_item) 
        user_behavior_cate = self.embedding(hist_cate)
        user_behavior = torch.cat([user_behavior_item, user_behavior_cate], dim = -1)
       
        # 6.矩阵相乘，将那些行为为空的地方全部写为0
        user_behavior = user_behavior.mul(mask)
        # 7.将用户行为乘上注意力系数,再把所有行为记录向量相加
        user_interest = self.AttentionActivate(item_embeddings, user_behavior, mask)
        
        # 8.将计算后的用户行为行为记录和推荐的目标进行拼接
        # inputs = torch.cat([user_embeddings, item_embeddings], dim = -1)
        inputs = torch.cat([user_embeddings, user_interest, item_embeddings], dim = -1)
        # 9.输入用户行为和目标向量，计算预测得分
        #print(item_embeddings)
        #print(item_embeddings.shape)
        #print(item_embeddings.dim())
        #print(user_embeddings.dim())
        out = self.mlp(inputs)
        # 10.sigmoid激活函数
        out = torch.sigmoid(out)        
        return out