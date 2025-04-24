import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True): #输入特征的数量，输出特征的数量
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self): # 使用从均匀分布中采样的值初始化权重和偏置参数
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor): # 接受输入特征x和图结构G
        # 输出x和self.weight的形状
        # print("Input x shape:", x.shape)  # 打印输入特征的形状
        # print("Weight shape:", self.weight.shape)  # 打印权重的形状

        x = x.matmul(self.weight) # 输入特征与权重矩阵相乘
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) # 将结果与图结构G相乘
        return x


# class HGNN_conv(nn.Module):
#     def __init__(self, in_ft, out_ft, drop_prob=0.2, bias=True):
#         """
#         超图卷积层
#         in_ft: 输入特征的维度
#         out_ft: 输出特征的维度
#         drop_prob: DropEdge 的概率
#         bias: 是否使用偏置
#         """
#         super(HGNN_conv, self).__init__()
#
#         # 定义卷积权重和偏置
#         self.weight = Parameter(torch.Tensor(in_ft, out_ft))
#         self.drop_prob = drop_prob  # DropEdge 的概率
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_ft))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """初始化权重和偏置"""
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def drop_edge(self, G):
#         """实现 DropEdge 的功能，随机丢弃 G 中的部分边"""
#         device = G.device
#         # 生成一个掩码，随机丢弃边
#         mask = (torch.rand_like(G) > self.drop_prob).float().to(device)
#         G = G * mask  # 用掩码矩阵对图矩阵 G 进行稀疏化处理
#         return G
#
#     def forward(self, x: torch.Tensor, G: torch.Tensor):
#         """
#         前向传播过程：
#         x: 输入特征矩阵
#         G: 超图的邻接矩阵
#         """
#         # 应用 DropEdge 随机丢弃部分边
#         G = self.drop_edge(G)
#
#         # 输入特征与权重矩阵相乘
#         x = x.matmul(self.weight)
#         if self.bias is not None:
#             x = x + self.bias
#
#         # 使用稀疏化后的图结构 G 进行卷积操作
#         x = G.matmul(x)
#         return x