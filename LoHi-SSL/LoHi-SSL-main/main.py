import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, \
    accuracy_score
import opt  # 必须导入
from utils import *
from encoder import *
from LowOrder import LowOrder
from data_loader import load_data
from HighOrder_model.loss import intra_cell_loss, inter_cell_loss
from HighOrder_model.H_model import HGNN_supervised, HGNN_unsupervised, neighbor_sampling, neighbor_concat, \
    generate_node_pair_sets
from multiomics_hypergraph_construction import load_feature_and_H

args = opt.args  # 直接获取参数对象

# ==========================================
# 2. 辅助类定义
# ==========================================

class MLPProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjection, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss


def evaluate_clustering(emb1, z1, z2, labels_true, n_clusters):
    """封装评估逻辑，强制 CPU 运行"""
    # 拼接
    features_tensor = torch.cat([emb1, z1, z2], dim=1)
    features = features_tensor.detach().cpu().numpy()

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(features)

    metrics = {
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'ACC': cluster_acc(labels_true, labels_pred),
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'AMI': adjusted_mutual_info_score(labels_true, labels_pred)
    }
    return metrics


# ==========================================
# 3. 训练函数
# ==========================================
def train(model1, model2, data_dict, args):
    fts, G, H_tri, H_bi, H_single, H_none, X1, A1, X2, A2 = data_dict['tensors']
    labels_true = data_dict['labels_true']

    # 强制 CPU
    device = torch.device('cpu')

    N = fts.shape[0]

    # 初始化优化器
    optimizer1 = Adam(model1.parameters(), lr=args.lr)
    optimizer2 = Adam(model2.parameters(), lr=args.lr)
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=args.milestones, gamma=args.gamma)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.milestones, gamma=args.gamma)

    # 损失函数
    instance_loss = InstanceLoss(batch_size=N, temperature=args.tau).to(device)
    criterion_intra = intra_cell_loss(args.beta)
    criterion_inter = inter_cell_loss(args.tau)
    projection_layer = MLPProjection(input_dim=args.n_hid, hidden_dim=64, output_dim=20).to(device)

    # 记录最佳指标 (metrics)
    metrics = {'NMI': 0, 'ACC': 0, 'ARI': 0, 'AMI': 0, 'Epoch': 0}

    # 记录最佳输出 (outputs)
    outputs = (None, None, None)

    print(f"Start training for {args.max_epoch} epochs on CPU (Silent Mode)...")
    model1.train()
    model2.train()

    iterator = range(args.max_epoch)

    for epoch in iterator:
        # 1. 邻居采样
        coor_sampled_tri = neighbor_sampling(H_tri, args.positive_neighbor_num, args.p_tri)
        coor_sampled_bi = neighbor_sampling(H_bi, args.positive_neighbor_num, args.p_bi)
        coor_sampled_single = neighbor_sampling(H_single, args.positive_neighbor_num, 1 - args.p_tri - args.p_bi)
        H_union = neighbor_concat(coor_sampled_tri, coor_sampled_bi, coor_sampled_single, N)
        # .copy() 避免内存不连续导致的 Windows 崩溃
        H_union = torch.from_numpy(H_union.copy()).to(device)

        # 2. 前向传播
        embedding1, x_pos, x_neg = model1(fts, G)
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, cons1, mu1, log_var1 = model2(X1, A1, X2, A2)

        # 投影
        embedding1_proj = projection_layer(embedding1)

        # 3. 计算损失
        loss_intra = criterion_intra(embedding1, x_pos, x_neg)
        loss_inter = criterion_inter(embedding1, H_union, H_none)

        l_con_1 = instance_loss(embedding1_proj, Z1)
        l_con_2 = instance_loss(embedding1_proj, Z2)
        l_con_3 = instance_loss(Z1, Z2)

        L_DRR = drr_loss(cons1)
        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)
        L_KL = kl_loss(mu1, log_var1)

        loss = (L_DRR +
                args.alpha1 * L_REC1 + args.alpha2 * L_REC2 +
                args.beta * loss_intra + args.tau * loss_inter +
                args.lambda1 * L_KL +
                args.lambda2 * l_con_1 + args.lambda3 * l_con_2 + args.lambda4 * l_con_3)

        # 4. 反向传播
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        scheduler1.step()
        scheduler2.step()

        # 5. 评估 (只在后台更新 metrics 和 outputs)
        if epoch % args.print_freq == 0 or epoch == args.max_epoch - 1:
            with torch.no_grad():
                curr_metrics = evaluate_clustering(embedding1_proj, Z1, Z2, labels_true, args.n_clusters)

                # 如果当前 ACC 更好，则更新
                if curr_metrics['ACC'] > metrics['ACC']:
                    metrics = curr_metrics.copy()
                    metrics['Epoch'] = epoch
                    # 更新 outputs
                    outputs = (embedding1_proj.clone(), Z1.clone(), Z2.clone())

    # 防止 outputs 为空
    if outputs[0] is None:
        outputs = (embedding1_proj, Z1, Z2)

    return metrics, outputs[0], outputs[1], outputs[2]


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 同步 args 到 opt
    opt.args = args

    device = torch.device('cpu')
    args.device = 'cpu'

    setup_seed(args.seed)

    print(f"------------------------------")
    print(f"Dataset: {args.dataset} | Device: {device}")
    print(f"------------------------------")

    # 1. 加载 HighOrder 数据
    fts, lbls, H, H_rna, H_atac = load_feature_and_H(
        args.data_dir_rna, args.data_dir_atac, args.lbls_dir,
        m_prob=args.m_prob, K_neigs=args.K_neigs, is_probH=args.is_probH,
        edge_type=args.edge_type, use_rna=args.use_rna, use_atac=args.use_atac
    )

    G = generate_G_from_H(H)
    H_tri, H_bi, H_single, H_none = generate_node_pair_sets(H_rna, H_atac)

    # 2. 处理标签
    if torch.is_tensor(lbls):
        labels_true = lbls.cpu().numpy()
    else:
        labels_true = lbls

    num_clusters = len(np.unique(labels_true))
    args.n_clusters = num_clusters
    opt.args.n_clusters = num_clusters

    # 3. 加载 LowOrder 数据
    X1, _, A1 = load_data(args.dataset, 'rna', args.method, args.k, show_details=False)
    X2, _, A2 = load_data(args.dataset, 'atac', args.method, args.k, show_details=False)

    # 4. 数据转 Tensor 并移至 CPU
    fts = torch.Tensor(fts).to(device)
    G = torch.Tensor(G).to(device)
    H_none = torch.from_numpy(H_none).to(device)
    X1 = torch.tensor(X1, dtype=torch.float32).to(device)
    A1 = torch.tensor(A1, dtype=torch.float32).to(device)
    X2 = torch.tensor(X2, dtype=torch.float32).to(device)
    A2 = torch.tensor(A2, dtype=torch.float32).to(device)

    # 5. 初始化模型 (全部 to device='cpu')
    n_d1 = X1.shape[1]
    n_d2 = X2.shape[1]

    ae1 = VAE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2,
              ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2,
              n_input=n_d1, n_z=args.n_z).to(device)
    ae2 = AE(ae_n_enc_1=args.ae_n_enc_1, ae_n_enc_2=args.ae_n_enc_2,
             ae_n_dec_1=args.ae_n_dec_1, ae_n_dec_2=args.ae_n_dec_2,
             n_input=n_d2, n_z=args.n_z).to(device)

    gae1 = IGAE(gae_n_enc_1=args.gae_n_enc_1, gae_n_enc_2=args.gae_n_enc_2,
                gae_n_dec_1=args.gae_n_dec_1, gae_n_dec_2=args.gae_n_dec_2,
                n_input=n_d1, n_z=args.n_z, dropout=args.dropout).to(device)
    gae2 = IGAE(gae_n_enc_1=args.gae_n_enc_1, gae_n_enc_2=args.gae_n_enc_2,
                gae_n_dec_1=args.gae_n_dec_1, gae_n_dec_2=args.gae_n_dec_2,
                n_input=n_d2, n_z=args.n_z, dropout=args.dropout).to(device)

    model_ft = HGNN_unsupervised(in_ch=fts.shape[1], n_hid=args.n_hid, dropout=args.dropout).to(device)
    model_LowOrder = LowOrder(ae1, ae2, gae1, gae2, n_node=X1.shape[0]).to(device)

    # 6. 开始训练
    data_dict = {
        'tensors': (fts, G, H_tri, H_bi, H_single, H_none, X1, A1, X2, A2),
        'labels_true': labels_true
    }

    t0 = time.time()
    best_results, emb1, z1, z2 = train(model_ft, model_LowOrder, data_dict, args)
    t1 = time.time()

    print("\n" + "=" * 40)
    print(f"Training Finished. Time cost: {t1 - t0:.2f}s")
    print(f"Best Result Found at Epoch {metrics['Epoch']}:")
    print(f"NMI: {metrics['NMI']:.4f}")
    print(f"ACC: {metrics['ACC']:.4f}")
    print(f"ARI: {metrics['ARI']:.4f}")
    print(f"AMI: {metrics['AMI']:.4f}")
    print("=" * 40)

    # 保存
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'embedding1_proj.npy'), emb1.cpu().detach().numpy())
    np.save(os.path.join(output_dir, 'Z1.npy'), z1.cpu().detach().numpy())
    np.save(os.path.join(output_dir, 'Z2.npy'), z2.cpu().detach().numpy())
