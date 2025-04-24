import torch
import random
from sklearn import metrics
from munkres import Munkres
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_mutual_info_score
import opt
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numpy_to_torch(a, sparse=False):
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


# the reconstruction function
# def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
#     loss_ae = F.mse_loss(X_hat, X)
#     loss_w = F.mse_loss(Z_hat, torch.spmm(A_norm, X))
#     loss_a = F.mse_loss(A_hat, A_norm.to_dense())
#     loss_igae = loss_w + opt.args.alpha_value * loss_a
#     loss_rec = loss_ae + loss_igae
#     return loss_rec

def reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat):
    # 计算重建损失
    loss_ae = F.mse_loss(X_hat, X)

    # 使用稀疏矩阵乘法计算
    if A_norm.is_sparse:
        # 确保 A_norm 是稀疏矩阵
        A_norm = A_norm.coalesce()  # 确保 A_norm 是稀疏矩阵且没有重复项
        loss_w = F.mse_loss(Z_hat, torch.sparse.mm(A_norm, X))  # 使用稀疏矩阵乘法
    else:
        loss_w = F.mse_loss(Z_hat, torch.mm(A_norm, X))  # 如果 A_norm 是密集矩阵

    # 对 A_hat 进行处理
    if A_hat.is_sparse:
        A_hat = A_hat.to_dense()  # 将 A_hat 转换为密集矩阵
    if A_norm.is_sparse:
        A_norm = A_norm.to_dense()  # 将 A_norm 转换为密集矩阵

    loss_a = F.mse_loss(A_hat, A_norm)

    # 综合损失
    loss_igae = loss_w + opt.args.alpha_value * loss_a
    loss_rec = loss_ae + loss_igae

    return loss_rec

def kl_loss(mu, log_var):
    """
    计算 KL 散度损失。

    Args:
        mu (Tensor): 从 VAE 编码器获得的均值。
        log_var (Tensor): 从 VAE 编码器获得的对数方差。

    Returns:
        Tensor: KL 散度损失。
    """
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_div

def target_distribution(Q):
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


# clustering guidance
def distribution_loss(Q, P):
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    # loss = F.kl_div(Q[0].log(), P, reduction='batchmean')
    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_correlation(Z_v1, Z_v2):

    return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t())


def correlation_reduction_loss(S):

    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()


def drr_loss(cons):

    S_N = cross_correlation(cons[0], cons[1])
    L_N = correlation_reduction_loss(S_N)

    S_F = cross_correlation(cons[2], cons[3])
    L_F = correlation_reduction_loss(S_F)

    loss_drr = opt.args.lambda1 * L_N + opt.args.lambda2 * L_F

    return loss_drr


def clustering(Z, y):
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=10)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())

    ari, nmi, ami, acc = eva(y, cluster_id, show_details=True)

    return ari, nmi, ami, acc, model.cluster_centers_


def assignment(Q, y):
    y_pred = torch.argmax(Q, dim=1).data.cpu().numpy()
    ari, nmi, ami, acc = eva(y, y_pred, show_details=False)
    return ari, nmi, ami, acc, y_pred


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0

    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)

    return acc


def eva(y_true, y_pred, show_details=True):

    acc = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    
    if show_details:
        print("\n","ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi), "AMI: {:.4f}".format(ami), "ACC: {:.4f},".format(acc))
        
    return ari, nmi, ami, acc


def Pear_corr(x):  # 计算给定数据的皮尔逊相关系数矩阵
    x = x.T
    x_pd = pd.DataFrame(x)
    dist_mat = x_pd.corr()
    return dist_mat.to_numpy()


def Eu_dis(x):  # 计算给定数据之间的欧氏距离矩阵
    dist_mat = cdist(x, x, 'euclid')
    print('dist_mat shape:', dist_mat.shape)
    return dist_mat


# Concat the features from multiple modalities 将多个模态的特征进行连接的函数
def Multi_omics_feature_concat(*F_list, normal_col=False):  # *F_list，表示多个特征矩阵

    features = None
    for f in F_list:  # 对于每个特征矩阵 f
        # if f is not None and f != []:  # 如果 f 不为空且不是空列表
        #     if len(f.shape) > 2:  # 如果特征矩阵 f 的维度大于 2，则将其展平为二维
        #         f = f.reshape(-1, f.shape[-1])
        #     if normal_col:  # 如果 normal_col 为真，则对 f 的每一列进行归一化处理
        #         f_max = np.max(np.abs(f), axis=0)
        #         f = f / f_max
        if features is None:  # 如果 features 是空的，将 f 赋值给 features
            features = f
        else:
            features = np.hstack((features, f))  # 如果 features 不为空，则将 f 水平连接到 features 后面
    if normal_col:  # 如果 normal_col 为真，则对整个特征矩阵进行归一化处理，并返回归一化后的特征矩阵
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


# Concat the modality-specific hyperedges  函数会遍历所有输入的超图关联矩阵，并将它们连接成一个整体的超图关联矩阵。
def Multi_omics_hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and (isinstance(h, list) or np.any(h)):
            if H is None:
                H = h if not isinstance(h, list) else [np.array(x) for x in h]
            else:
                if not isinstance(h, list):
                    H = np.hstack((H, h))
                else:
                    H = [np.hstack(pair) for pair in zip(H, h)]
    if isinstance(H, list):
        H = np.hstack(H)
    return H


# generate G from incidence matrix H   设计目的是从关联矩阵（H）生成图（G）
def generate_G_from_H(H, variable_weight=False):
    H = np.array(H)  # 转换为 NumPy 数
    n_edge = H.shape[1]  # n_edge 被确定为 H 中的列数，表示超边的数量
    # the weight of the hyperedge, here we use 1 for each hyperedge W 被初始化为全为1的数组，表示每个超边的权重相等
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # 通过沿着行对 H 和 W 的乘积进行求和来计算每个节点的度
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # 通过沿着列对 H 进行求和来计算每个超边的度

    invDE = np.mat(np.diag(np.power(DE, -1)))  # invDE 计算 DE 的逆矩阵
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  # DV2 计算 DV 的平方根作为对角矩阵
    W = np.mat(np.diag(W))  # W 也被转换为对角矩阵
    H = np.mat(H)
    HT = H.T

    if variable_weight:  # 如果 variable_weight 是 True
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2  # 则返回中间矩阵（DV2_H、W、invDE_HT_DV2）供进一步计算使用
    else:  # 如果 variable_weight 是 False，如下公式计算图
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, edge_type='euclid'):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))  # 初始化一个全零矩阵 H

    if edge_type == 'euclid':  # 对于欧氏距离 (edge_type == 'euclid')
        for center_idx in range(n_obj):  # 遍历每个对象的索引
            dis_mat[center_idx, center_idx] = 0  # 将每个对象与自身的距离设置为0
            dis_vec = dis_mat[center_idx]  # 获取当前对象与其他对象的距离向量

            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()  # 根据距离向量排序得到距离最近的对象索引
            avg_dis = np.average(dis_vec)  # 计算距离向量的平均值，作为后续计算中的参考值
            if not np.any(nearest_idx[:k_neig] == center_idx):  # 如果当前对象不在最近的 k_neig 个对象中，将其加入到最近的对象中
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:  # 遍历最近的 k_neig 个对象
                if is_probH:  # is_probH 为 True，即需要将连接权重转化为概率
                    H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)

                else:  # 如果 is_probH 为 False，即不需要转化为概率
                    H[node_idx, center_idx] = 1.0
        print('use euclid for H construction')

    elif edge_type == 'pearson':  # 当前正在处理使用皮尔逊相关系数进行构建连接关系
        for center_idx in range(n_obj):  # 遍历每个对象的索引
            dis_mat[center_idx, center_idx] = -999.  # 将对象与自身的相关性设定为一个非常小的值
            dis_vec = dis_mat[center_idx]  # 获取当前对象与其他对象的相关性向量
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()  # 根据相关性向量排序得到相关性最大的对象索引
            nearest_idx = nearest_idx[::-1]  # 由于皮尔逊相关系数越大表示相关性越强，所以我们将排序结果反转，以便得到相关性最大的对象索引

            avg_dis = np.average(dis_vec)  # 计算相关性向量的平均值
            if not np.any(nearest_idx[:k_neig] == center_idx):  # 如果当前对象不在相关性最大的 k_neig 个对象中，将其加入到相关性最大的对象中
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:  # 遍历相关性最大的 k_neig 个对象
                if is_probH:  # 需要将连接权重转化为概率
                    H[node_idx, center_idx] = 1. - np.exp(-(dis_vec[node_idx] + 1.0) ** 2)
                else:
                    H[node_idx, center_idx] = 1.0
        print('use pearson for H construction')

    return H


def construct_H_with_KNN(X, K_neigs=[10], is_probH=True, m_prob=1, edge_type='euclid'):
    if len(X.shape) != 2:  # 如果输入数据 X 的维度不是二维的，则将其转为二维
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:  # 若K_neigs不是列表将其转换为包含该整数的列表
        K_neigs = [K_neigs]

    if edge_type == 'euclid':
        dis_mat = Eu_dis(X)

    elif edge_type == 'pearson':
        dis_mat = Pear_corr(X)

    # H = []
    H = np.zeros((X.shape[0], X.shape[0]))  # 初始化一个全零矩阵 H
    for k_neig in K_neigs:  # 对于每个指定的最近邻数量
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob, edge_type)
        H = Multi_omics_hyperedge_concat(H, H_tmp)

    return H


def sample_(label, y):
    idx = np.where(y == label)[0]  # 找到标签数组 y 中与指定标签值 label 相匹配的索引
    return np.random.choice(idx)  # 从匹配的索引中随机选择一个，并返回该索引
