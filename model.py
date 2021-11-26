# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import Get_rel_dic

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


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        x = x.permute(0, 3, 1, 2)
        out = [x]
        support = [support]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h.permute(0,2,3,1)


class Global(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(Global, self).__init__()
        # Spatial Embedding
        self.adj = adj

        self.gcnF = gcn(embed_size,embed_size,dropout = dropout,support_len=1,order=2)
        self.gcnD = gcn(embed_size, embed_size, dropout=dropout, support_len=1, order=2)

        self.Att1 = SMultiHeadAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )


        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 6),
            nn.Linear(embed_size * 6, embed_size)
        )

    def forward(self,query):
        B, N, T, H = query.shape

        # GCN 部分
        X_F = self.gcnF(query, self.adj.to(device))


        R = query.reshape(B, N, T*H)
        R = self.ff(R)
        R = R.sum(dim=0)
        A = torch.matmul(R, R.transpose(-1, -2))
        R = torch.relu(torch.softmax(A, -1)) + torch.eye(A.shape[1]).to(device)


        X_D = self.gcnD(query, R)

        X_DF = torch.tanh(X_D)*torch.sigmoid(X_F)

        #query = self.norm1(query + X_DF )
        attention = self.Att1(X_DF, X_DF, X_DF)  # (B, N, T, C)
        x = self.dropout(self.norm2(attention + X_DF))
        forward = self.feed_forward(x)
        X_DF  = self.dropout(self.norm3(forward + x))


        return X_DF  # (B, N, T, C)


class Local(nn.Module):
    def __init__(self, embed_size, heads,innode,outnode, dropout, forward_expansion):
        super(Local, self).__init__()
        self.miRNA = outnode
        self.mRNA = innode-outnode
        self.rel_dic = Get_rel_dic(self.mRNA,self.miRNA)
        self.conv1 = nn.Conv2d(1,embed_size,1,1)
        self.conv2 = nn.Conv2d(embed_size,embed_size,1,1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1, 1)
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    def forward(self,x):
        D = torch.zeros((x.shape[0], self.miRNA, 100)).to(device)
        for index in range(self.mRNA,self.mRNA+self.miRNA):
            temp = x[:,self.rel_dic[index]]
            B,N = temp.shape
            D[0:B, index - self.mRNA, 0:N] += temp

        x = D.unsqueeze(-1).permute(0,3,1,2)

        x = self.conv1(x)
        x = self.conv2(x).permute(0,2,3,1)

        x_a = self.attention(x,x,x)

        x_a = self.dropout(self.norm1(x_a+x))
        feed = self.feed_forward(x_a)
        x = self.dropout(self.norm2(feed+x_a)).permute(0,3,1,2)

        x = self.conv3(x).squeeze(1)

        x = self.out(x).squeeze(-1)

        return x


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
        self.innode = innode
        self.outnode = outnode
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
        #embed_size, heads, adj, dropout, forward_expansion
        self.Global1 = Global(embed_size,heads,adj,dropout,forward_expansion)
        self.Global2 = Global(embed_size,heads,adj,dropout,forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        #embed_size, heads, adj, dropout, forward_expansio
        self.Local = Local(embed_size,heads,innode,outnode,dropout,forward_expansion)

        self.fc1 = nn.Sequential(
            nn.Linear(innode,512),
            nn.ReLU(),
            nn.Linear(512,70),
                                 )

        self.fs = nn.Linear(outnode, outnode)
        self.fg = nn.Linear(outnode, outnode)

    def forward(self, x):
        x_L = x[:,0:self.innode-self.outnode]
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1).unsqueeze(1)
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        out = self.Global1(input_Transformer)
        out = self.norm1(out+input_Transformer)
        out = self.Global2(out)
        L = self.Local(x_L)
        #####################################
        out = out.permute(0, 3, 1, 2)
        out = self.relu(self.conv2(out)).squeeze(-1).squeeze(1)

        out = self.fc1(out)

        g = torch.sigmoid(self.fg(out)+self.fs(L))

        out = g*out+(1-g)*L

        return out


#####################################
#adj = torch.randn(200,200)
# model = STTF(adj,200,100,1,64,4,4,0)
#
# x = torch.randn(1,200)
# print(model(x))
#embed_size, heads, adj, dropout, forward_expansion):
# L = Local(64,4,adj,0,4)
# x = torch.randn(2,4880)
# print(x.shape)
# M = L(x)

