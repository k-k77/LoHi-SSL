import torch
from torch import nn
import torch.nn.functional as F


# dual contrastive loss function

# contrastive loss1: intra-cell contrastive loss
# Takes embeddings of an anchor sample, a positive sample and a negative sample
class intra_cell_loss(nn.Module):

    def __init__(self, margin):
        super(intra_cell_loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5) 计算了 anchor 到 positive、negative 的欧氏距离的平方和
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin) # 通过比较它与正样本和负样本之间的距离来计算损失
        return losses.mean() if size_average else losses.sum()


# contrastive loss2: inter-cell contrastive loss
# pull similar node pairs closer and push dissimilar node pairs apart
class inter_cell_loss(nn.Module):

    def __init__(self,tau):
        super(inter_cell_loss,self).__init__()
        self.tau = tau  # tau：损失函数中的温度参数

    def sim(self, x_ach: torch.Tensor):  # 相似度计算
        x_ach = F.normalize(x_ach)    # 对输入向量进行 L2 归一化，以确保每个向量的模长为 1
        return torch.mm(x_ach, x_ach.t())  # 每对样本之间的内积
  
    def forward(self, x_ach: torch.Tensor, H_union,H_none):
        f = lambda x: torch.exp(x / self.tau)
        sim_mat = f(self.sim(x_ach))  # 调用 sim 方法计算样本嵌入向量之间的相似度矩阵，然后将相似度矩阵转换为相似度得分
        neighbor_sim_mat = torch.mul(H_union,sim_mat)  # 将相似度得分与 H_union 和 H_none 相乘
        none_neighbor_sim_mat = torch.mul(H_none,sim_mat)
        loss = -torch.log(neighbor_sim_mat.sum()/(none_neighbor_sim_mat.sum()))
        return loss




