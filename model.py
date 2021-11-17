# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# from main import device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 用Linear来做投影矩阵
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = adj.to(device)
        # elf.D_S = adj
        self.embed_liner = nn.Linear(adj.shape[0], embed_size)

        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.norm5 = nn.LayerNorm(embed_size)
        self.norm6 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        # 调用GCN
        self.norm_adj = nn.InstanceNorm2d(1)  # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

        self.ff1 = nn.Linear(embed_size, embed_size)
        self.ff2 = nn.Linear(embed_size, embed_size)

        self.E = nn.Sequential(
            nn.Linear(adj.shape[0], embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
        #self.Att = Attention(d_model=embed_size * 12, d_k=embed_size * 2, d_v=embed_size * 2, h=heads, dropout=0)
        self.Att = SMultiHeadAttention(embed_size, heads)
    def forward(self, value, key, query):
        B, N, T, C = query.shape
        query1 = query
        # GCN 部分
        X_G = torch.Tensor(B, N, 0, C).to(device)
        # self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        # self.adj = self.norm_adj(self.adj)
        # self.adj = self.adj.squeeze(0).squeeze(0)

        for t in range(query.shape[2]):
            # o = self.gcn(query[:, :, t, :], self.adj)  # [B, N, C]
            # o = o.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
            # #             print(o.shape)
            # X_G = torch.cat((X_G, o), dim=2)
            x = query[:, :, t, :]
            A = self.adj.to(device)
            D = (A.sum(-1) ** -0.5)
            D[torch.isinf(D)] = 0.
            D = torch.diag_embed(D)
            A = torch.matmul(torch.matmul(D, A), D)
            x = torch.relu(self.ff1(torch.matmul(A, x)))
            # x = torch.softmax(self.ff2(torch.matmul(A, x)), dim=-1)
            x = x.unsqueeze(2)
            X_G = torch.cat((X_G, x), dim=2)
        # 最后X_G [B, N, T, C]

        #         print('After GCN:')
        #         print(X_G)
        # Spatial Transformer 部分
        X_E = self.E(self.adj.to(device)).reshape(N, T, C).unsqueeze(0)
        query = self.norm5(query + X_E)
        attention = self.attention(query, query, query)  # (B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        X_G = self.norm6(query1+X_G)
        #Attention1 = self.Att(X_G.reshape(B, N, -1), X_G.reshape(B, N, -1), X_G.reshape(B, N, -1)).reshape(-1, N, T, C)
        Attention1 = self.Att(X_G,X_G,X_G)
        y = self.dropout(self.norm3(Attention1 + X_G))
        forward1 = self.feed_forward1(y)
        X_G = self.dropout(self.norm4(y + forward1))
        # 融合 STransformer and GCN



        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))  # (7)
        out = g * U_S + (1 - g) * X_G  # (8)

        return out  # (B, N, T, C)





class TBlock(nn.Module):
    def __init__(
            self,
            adj,
            in_channels,
            embed_size,
            heads,
            forward_expansion,
            dropout=0

    ):
        super(TBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv4 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv5 = nn.Conv2d(in_channels, embed_size, 1)
        # embed_size, heads, adj, dropout, forward_expansion
        self.S = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)

        # 缩小时间维度
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        input_Transformer = x

        # 位置潜入
        B, N, T, H = input_Transformer.shape
        # input_Transformer = self.positional_encoding(input_Transformer)
        # **********************************************************************************************
        # 提取空间信息
        output_S = self.S(input_Transformer, input_Transformer, input_Transformer)
        # 残差+层归一化
        out = self.norm1(output_S + input_Transformer)
        # **********************************************************************************************


        return out


class STTF(nn.Module):
    def __init__(
            self,
            adj,
            innode,
            outnode,
            in_channels,
            embed_size,
            heads,
            forward_expansion,
            dropout=0

    ):
        super(STTF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv2 = nn.Conv2d(innode, outnode, 1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        self.ST1 = TBlock(adj,in_channels,embed_size,heads,forward_expansion,dropout=dropout)
        self.ST2 = TBlock(adj, in_channels, embed_size, heads,forward_expansion, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1).unsqueeze(1)
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        out = self.ST1(input_Transformer)
        out = self.norm1(out+input_Transformer)
        out = self.ST2(out)

        #####################################
        #out = out.permute(0, 2, 1, 3)
        out = self.relu(self.conv2(out))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.squeeze(1)
        return out.permute(0, 2, 1).squeeze(-1)


#####################################
# adj = torch.randn(200,200)
# model = STTF(adj,200,100,1,64,4,4,0)
#
# x = torch.randn(1,200)
# print(model(x))