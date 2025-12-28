import torch
from torch import nn
from .layer import HGNN_conv
import torch.nn.functional as F
import random
from scipy.sparse import coo_matrix
import numpy as np



# Xavier初始化
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# 定义线性层，带有Xavier初始化
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class HGNN_unsupervised(nn.Module):
    def __init__(self, in_ch, n_hid, dropout):  # 输入特征的通道数、隐藏层的大小、Dropout 概率
        super(HGNN_unsupervised, self).__init__()
        self.dropout = dropout

        # 初始化特征信息编码器
        self.FeatureInforEncoder = LinearLayer(in_ch, in_ch)
        self.FeatureEncoder = LinearLayer(in_ch, 200)

        # define HGNN encoder 定义了两个 HGNN 编码器
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

        # define mlp encoder 定义两个 MLP 编码器
        self.mlp1 = nn.Linear(in_ch, n_hid)
        self.mlp2 = nn.Linear(n_hid, n_hid)

        


    def forward(self, x, G):  # x 是输入特征，G 是图结构
        # 通过特征信息编码器计算特征信息
        feature_info = torch.sigmoid(self.FeatureInforEncoder(x))

        # 使用特征信息增强输入特征
        x_enhanced = x * feature_info

        # 进一步提取隐藏特征
        # x_hidden = F.relu(self.FeatureEncoder(x_enhanced))
        x_hidden = F.gelu(self.FeatureEncoder(x_enhanced))
        x_hidden = F.dropout(x_hidden, self.dropout, training=self.training)

        # 对 x 进行特征信息计算增强
        feature_info_pos = torch.sigmoid(self.FeatureInforEncoder(x))
        x_pos_enhanced = x * feature_info_pos  # 增强 x_pos 特征

        
        x_pos = F.gelu(self.mlp1(x_pos_enhanced))
        x_pos = F.dropout(x_pos, self.dropout)
        x_pos = self.mlp2(x_pos)
       

        # HGNN 编码
        # x_ach = F.relu(self.hgc1(x_hidden, G))
        x_ach = F.gelu(self.hgc1(x_hidden, G))
        x_ach = F.dropout(x_ach, self.dropout)
        x_ach = self.hgc2(x_ach, G)
      

        x_neg = x_pos[torch.randperm(x_pos.size()[0])]  # 从正样本表示 x_pos 中随机抽取一个样本作为负样本表示 x_neg
        return x_ach, x_pos, x_neg



class HGNN_supervised(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_supervised, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):  # x 是输入特征，G 是图结构
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        x = F.dropout(x, self.dropout)  # 在每个 ReLU 激活后都应用 Dropout，最后一层除外
        x = self.hgc3(x, G)
        return x


def generate_node_pair_sets(H_rna, H_atac):  #对输入的图进行预处理
        

    np.fill_diagonal(H_rna, 0)
    # np.fill_diagonal(H_adt, 0)
    np.fill_diagonal(H_atac, 0)


    H_rna = np.where(H_rna,1,0)
    # H_adt = np.where(H_adt,1,0)
    H_atac = np.where(H_atac,1,0)
    # H_all = H_rna + H_adt + H_atac
    H_all = H_rna + H_atac
    H_tri = np.where(H_all==3,1,0)
    H_bi = np.where(H_all==2,1,0)
    H_single = np.where(H_all==1,1,0)
    H_none = np.where(H_all==0,1,0)



    return H_tri, H_bi, H_single, H_none




def neighbor_sampling(H,positive_neighbor_num,p):  
    row_coor, col_coor = np.nonzero(H)  # 找到关联矩阵中非零元素的行坐标和列坐标
    coor = np.vstack((row_coor,col_coor))
    indices = list(range(coor.shape[1]))  # 将它们堆叠成一个坐标数组
    random.shuffle(indices)  # 随机打乱坐标数组的索引
    num_subset = int(positive_neighbor_num*p)
    idx_subset = indices[:num_subset]
    coor_sampled = coor[:,idx_subset]

    return coor_sampled  # 最后，返回采样后的坐标数组


def neighbor_concat(coor_sampled_tri,coor_sampled_bi,coor_sampled_single,N):
    
    
    coor = np.hstack((coor_sampled_tri,coor_sampled_bi,coor_sampled_single))  # 将三种类型的采样邻居坐标数组水平堆叠在一起，形成一个大的坐标数组
    data = np.ones(coor.shape[1])  # 为每个坐标点设置一个值为1的数据
    return coo_matrix((data,(coor[0,:],coor[1,:])),shape=(N,N)).toarray()  # 利用 coo_matrix 函数将坐标数组转换成稀疏矩阵，并将其转换为密集矩阵（数组）返回
