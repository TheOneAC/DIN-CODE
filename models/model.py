
import torch
import torch.nn as nn
from models.layers import MLP
from models.din import DinAttentionLayer
from models.dien import DienAttentionLayer
from models.dicn import DicnAttentionLayer

SEQ_ATTENTION = {
    'DIN': lambda item_att_dim, drop, hist_len :DinAttentionLayer(item_att_dim, drop, hist_len),
    'DIEN': lambda item_att_dim, drop, hist_len :DienAttentionLayer(item_att_dim, drop, hist_len),
    'DICN': lambda item_att_dim, drop, hist_len :DicnAttentionLayer(item_att_dim, drop, hist_len),
}


class Model(nn.Module):
    """
      模型主体
    """

    def __init__(self, feature_dim, embed_dim, feat_num, mlp_layers, dropout, hist_len, name='DIN'):
        super(Model, self).__init__()
        # 1.特征维度，就是输入的特征有多少个类
        self.feature_dim = feature_dim
        # 2.embeding层，将特征数值转化为向量
        self.embedding = nn.Embedding(feature_dim+1, embed_dim)
        # 3.注意力计算层（论文核心）
        user_feat_num, hist_feat_num,  item_feat_num = feat_num
        self.item_att_dim = embed_dim*hist_feat_num
        self.AttentionActivate = SEQ_ATTENTION[name](self.item_att_dim, dropout, hist_len)
        self.input_dim = (user_feat_num + item_feat_num)* embed_dim + self.item_att_dim
        self.mlp = MLP(mlp_layers, input_dim=self.input_dim, dropout = dropout)
    
    def forward(self, x):
        """
            x输入(behaviors*40,ads*1) ->（输入维度） batch*(behaviors+ads)
            
        """
        # 1.排除掉推荐目标
        user, item_feat, hist_item, hist_cate = x
        user = user.int().cpu()
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