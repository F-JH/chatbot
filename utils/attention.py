import torch
import numpy as np
from torch import nn

class ScaledDotProductAttention(nn.Module):
    '''
        计算Attention(Qi, Ki, Vi)
    '''
    def __init__(self, dropout=0.3):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, Q, K, V, attn_mask, scale):
        '''
        :param Q: [batch_size, n_head, m_q, d_head]
        :param K: [batch_size, n_head, m_k, d_head]
        :param V: [batch_size, n_head, m_v, d_head]
        :param scale: 1 / sqrt(d_head)
        :param attn_mask: mask [batch_size, 1, m_q, m_k]
        :return:
        '''
        attention = torch.matmul(Q, K.transpose(2, 3))  # [batch_size, n_head, m_q, m_k]
        attention = attention * scale
        attention = attention.masked_fill(attn_mask, -1e9)
        attention = torch.matmul(self.softmax(attention), V)    # [batch_size, n_head, m_q, d_head]
        attention = self.dropout(attention)
        return attention

class mutliHeadAttention(nn.Module):
    '''
        mutli head Attention
    '''
    def __init__(self, d_model, dim_feedforward, n_head, device, dropout=0.3):
        assert  dim_feedforward % n_head == 0
        super(mutliHeadAttention, self).__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.d_head = dim_feedforward // n_head

        self.scaledDotProductAttention = ScaledDotProductAttention()
        self.layerNorm = nn.LayerNorm(d_model, device=device)

        self.w_q = nn.Linear(d_model, dim_feedforward, bias=False, device=device)
        self.w_k = nn.Linear(d_model, dim_feedforward, bias=False, device=device)
        self.w_v = nn.Linear(d_model, dim_feedforward, bias=False, device=device)
        self.w_o = nn.Linear(dim_feedforward, d_model, bias=False, device=device)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        :param input_Q: [batch_size, m, d_model]
        :param input_K: [batch_size, m, d_model]
        :param input_V: [batch_size, m, d_model]
        :param attn_mask: function of attention mask
        :return: Add & Layer Norm [batch_size, m, d_model]
        '''
        batch_size = input_Q.size(0)
        # 计算Q K V可以串在一起，但需要分开计算attention dim_feedforward = sum(self.d_head)
        Q = self.w_q(input_Q)
        K = self.w_k(input_K)
        V = self.w_v(input_V)
        # 对Q,K,V整形，使其变成(batch_size, num_head, m, d_head)
        Q = Q.view(batch_size, -1, self.num_head, self.d_head).transpose(1,2)
        K = K.view(batch_size, -1, self.num_head, self.d_head).transpose(1,2)
        V = V.view(batch_size, -1, self.num_head, self.d_head).transpose(1,2)
        # 计算ScaledDotProductAttention       [batch_size, n_head, m_q, d_head]
        attention = self.scaledDotProductAttention(Q, K, V, attn_mask, torch.sqrt(torch.tensor(1.0 / self.d_head)))
        output = attention.transpose(1, 2).reshape(batch_size, -1, self.dim_feedforward)
        output = self.w_o(output)
        output = self.layerNorm(output + input_Q)   # [batch_size, m_q, d_model]
        return output