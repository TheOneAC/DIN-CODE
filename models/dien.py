
import torch
import torch.nn as nn
from models.layers import AUGRUCell, DynamicGRU, ActivationUnit


class DienAttentionLayer( nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.gru_based_layer = nn.GRU( embedding_dim * 2 , embedding_dim * 2, batch_first = True)
        self.attention_layer = ActivationUnit(hist_embedding_dim = embedding_dim, dropout = dropout)
        self.gru_customized_layer = DynamicGRU( embedding_dim * 2, embedding_dim * 2)


    def forward(self, query_ad, user_behavior, mask):
        seq_len = user_behavior.shape[1]
        output_based_gru, _ = self.gru_based_layer(user_behavior)
        attention_scores = self.attention_layer(query_ad, output_based_gru, mask, return_scores = True)
        output_customized_gru = self.gru_customized_layer(output_based_gru, attention_scores)
        attention_feature = output_customized_gru[range(len(seq_len)),seq_len - 1]
        return attention_feature 