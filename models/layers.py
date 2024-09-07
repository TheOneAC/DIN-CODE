import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
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
    def __init__(self, hist_embedding_dim, dropout=0.2, fc_dims = [32, 16]):
        super(ActivationUnit, self).__init__()
        # 1.初始化fc层
        fc_layers = []
        # 2.输入特征维度
        input_dim = hist_embedding_dim*4     
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
        #queries = torch.cat([query] * seq_len, dim=1)
        queries = torch.cat([query.unsqueeze(1)] * seq_len, dim=1)
        # 3.前面的把四个embedding合并成一个（4*embedding）的向量，
        #  第一个向量是目标商品的向量，第二个向量是用户行为的向量，
        #  至于第三个和第四个则是他们的相减和相乘（这里猜测是为了添加一点非线性数据用于全连接层，充分训练）
        attn_input = torch.cat([queries, user_behavior, queries - user_behavior, 
                                queries * user_behavior], dim = -1)
        out = self.fc(attn_input)
        return out

class MLP( nn.Module):
    
    def __init__(self, hidden_size, use_bias=True, dropout=0.1, activation_layer = 'ReLU', input_dim = 24):
        super().__init__()
        Activation = nn.ReLU
        dimension_pair = [input_dim] + hidden_size
        def _dense( in_dim, out_dim, bias = use_bias, activation = Activation, dropout=dropout):
            return nn.Sequential(
                nn.Linear( in_dim, out_dim, bias = bias),
                Activation(),
                nn.Dropout(p = dropout))
        
        layers = [_dense( dimension_pair[i], dimension_pair[i+1]) for i in range( len( hidden_size))]
        #layers.append(nn.Linear(hidden_size[-1], 1))
        self.model = nn.Sequential( *layers )
    
    def forward( self, X): return self.model(X)

class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True):
        super(AUGRUCell, self).__init__()

        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.update_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Tanh())


    def forward(self, X, h_prev, attention_score):
        temp_input = torch.cat( [ h_prev, X ] , dim = -1)
        r = self.reset_gate( temp_input)
        u = self.update_gate( temp_input)

        h_hat = self.h_hat_gate( torch.cat( [ h_prev * r, X], dim = -1) )

        u = attention_score.unsqueeze(1) * u
        h_cur = (1. - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = AUGRUCell( input_dim, hidden_dim, bias = True)

    def forward(self, X, attenion_scores , h0 = None ):
        B, T, D = X.shape
        H = self.hidden_dim
        
        output = torch.zeros( B, T, H ).type( X.type() )
        h_prev = torch.zeros( B, H ).type( X.type() ) if h0 == None else h0
        for t in range( T): 
            h_prev = output[ : , t, :] = self.rnn_cell( X[ : , t, :], h_prev, attenion_scores[ :, t] )
        return output