# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.layers import MLP, ActivationUnit

class DicnAttentionLayer(nn.Module):
    """
      注意力序列层
      功能是计算用户行为与预测目标之间的系数，并将所有的向量进行相加，这里的目的是计算出用户的兴趣的能力向量
    """
    def __init__(self, hist_embedding_dim,  dropout):
        super(DicnAttentionLayer, self).__init__()
        self.active_unit = ActivationUnit(hist_embedding_dim = hist_embedding_dim, 
                                          dropout = dropout)
        self.context_unit = MLP([16,1], hist_embedding_dim*2, dropout)
        
    def forward(self, query_ad, user_behavior, mask):
        """
          :param query_ad:targe目标x的embedding   -> （输入维度） batch*1*embed
          :param user_behavior:行为特征矩阵     -> （输入维度） batch*seq_len*embed
          :param mask:被padding为0的行为置为false  -> （输入维度） batch*seq_len*1
          :return output:用户行为向量之和，反应用户的爱好
        """
        seq_len = user_behavior.shape[1]
        global_context = nn.AvgPool1d(kernel_size=seq_len)(user_behavior)
        local_context = user_behavior[:,-1,:] 
        context = torch.cat([global_context, local_context], dim=1)
        intention = self.context_unit(context)
        user_behavior = user_behavior * intention
        attns = self.active_unit(query_ad, user_behavior)     
        output = user_behavior.mul(attns.mul(mask))
        output = user_behavior.sum(dim=1)
        return output