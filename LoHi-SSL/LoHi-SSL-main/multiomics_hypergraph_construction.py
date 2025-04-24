from utils import *
import numpy as np
import torch
import torch.nn as nn

# Xavier 初始化函数
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearLayer, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)  # 应用 Xavier 初始化

    def forward(self, x):
        return self.clf(x)

class FeatureInfoExtractor(nn.Module):
    def __init__(self, in_dim):
        super(FeatureInfoExtractor, self).__init__()
        self.FeatureInforEncoder = LinearLayer(in_dim, in_dim)
        self.apply(xavier_init)  # 在整个模块上应用 Xavier 初始化

    def forward(self, x):
        feature_info = torch.sigmoid(self.FeatureInforEncoder(x))
        x_enhanced = x * feature_info
        return x_enhanced

# load RNA modality data
def load_rna_data(data_dir,lbls_dir):  # data_dir 是 RNA 数据的路径，lbls_dir 是标签数据的路径
    fts = np.load(data_dir)
    print('rna fts shape:',fts.shape)  # fts.shape指RNA 数据的形状
    lbls = np.load(lbls_dir)
    return fts, lbls

# # 数据加载和特征提取函数
# def load_rna_data(data_dir, lbls_dir):
#     fts = np.load(data_dir)
#     lbls = np.load(lbls_dir)
#
#     print('Original RNA feature shape:', fts.shape)  # RNA 数据的形状
#
#     # 将特征转换为 PyTorch 张量并进行特征提取
#     fts_tensor = torch.tensor(fts, dtype=torch.float32)
#     feature_extractor = FeatureInfoExtractor(fts.shape[1])
#     fts = feature_extractor(fts_tensor)  # 使用相同的变量名 fts 来存储增强后的数据
#
#     print("Enhanced RNA feature shape:", fts.shape)
#
#     return fts.detach().numpy(), lbls

# # load Protein modality data
# def load_adt_data(data_dir):
#     fts = np.load(data_dir)
#     print('protein fts shape:',fts.shape)
#     return fts


# load ATAC modality data
def load_atac_data(data_dir):
    fts = np.load(data_dir)
    print('atac fts shape:',fts.shape)
    return fts
# load ATAC modality data with feature extraction
# def load_atac_data(data_dir):
#     fts = np.load(data_dir)
#     print('Original ATAC feature shape:', fts.shape)  # ATAC 数据的形状
#
#     # 将特征转换为 PyTorch 张量并进行特征提取
#     fts_tensor = torch.tensor(fts, dtype=torch.float32)
#     feature_extractor = FeatureInfoExtractor(fts.shape[1])
#     fts = feature_extractor(fts_tensor)  # 使用相同的变量名 fts 来存储增强后的数据
#
#     print("Enhanced ATAC feature shape:", fts.shape)
#
#     return fts.detach().numpy()


def load_feature_and_H(data_dir_rna,

                     data_dir_atac,
                     lbls_dir,
                     m_prob=1,
                     K_neigs=[10],
                     is_probH=True,
                     edge_type = 'pearson',
                     use_rna = True,

                     use_atac = True):


    if use_rna:
        ft_rna, lbls = load_rna_data(data_dir_rna, lbls_dir)

    # if use_adt:
    #     ft_adt = load_adt_data(data_dir_adt)

    if use_atac:
        ft_atac = load_atac_data(data_dir_atac)

    fts = None
    if use_rna:
        fts = Multi_omics_feature_concat(fts, ft_rna)
    # if use_adt:
    #     fts = Multi_omics_feature_concat(fts, ft_adt)
    if use_atac:
        fts = Multi_omics_feature_concat(fts, ft_atac)
    print('fts shape:',fts.shape)  # 将这些数据连接起来成为一个整体特征矩阵 fts


    print('Constructing the multi-omics hypergraph incidence matrix!')
    H = None
    # 分别对 RNA、ADT 和 ATAC 数据构建 K 最近邻的超图邻接矩阵
    if use_rna:
        H_rna = construct_H_with_KNN(ft_rna, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
        H = Multi_omics_hyperedge_concat(H, H_rna)

    
    # if use_adt:
    #     H_adt = construct_H_with_KNN(ft_adt, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
    #     H = Multi_omics_hyperedge_concat(H, H_adt)

    
    if use_atac:
        H_atac = construct_H_with_KNN(ft_atac, K_neigs=K_neigs,is_probH=is_probH, m_prob=m_prob, edge_type=edge_type)
        H = Multi_omics_hyperedge_concat(H, H_atac)  # 将构建的邻接矩阵连接成为一个整体的超图关联矩阵


    print('Finish the Construction of hypergraph incidence matrix!')
    

    return fts, lbls, H, H_rna,  H_atac



