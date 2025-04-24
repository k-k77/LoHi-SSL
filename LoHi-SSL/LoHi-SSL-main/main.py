import tqdm
from torch.optim import Adam
from time import time
import time
from utils import *
from encoder import *
from LowOrder import scMIC
from data_loader import load_data
import os
import torch
import torch.optim as optim
from utils import generate_G_from_H, sample_
from HighOrder_model.loss import intra_cell_loss, inter_cell_loss
from HighOrder_model.H_model import HGNN_supervised, HGNN_unsupervised, neighbor_sampling, neighbor_concat, generate_node_pair_sets
from multiomics_hypergraph_construction import load_feature_and_H
import argparse
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, accuracy_score
from utils import *
parser = argparse.ArgumentParser()

# 参数设置
parser.add_argument("--dataset", type=str, default='sim1', help='Ma-2020-4 name')
parser.add_argument('--data_dir_rna', default='./input/Ma-2020-4/rna.npy', help='path of RNA data')
parser.add_argument('--data_dir_adt', default='./input/Ma-2020-4/adt.npy', help='path of Protein data')
parser.add_argument('--data_dir_atac', default='./input/Ma-2020-4/atac.npy', help='path of ATAC data')
parser.add_argument('--lbls_dir', default='./input/Ma-2020-4/lbls.npy', help='path of cell lbls')
parser.add_argument("--supervised", type=bool, default=False, help='True for stage 2 (cell type annotation), False for stage1 (unsupervised cell representation learning)')
parser.add_argument("--m_prob", type=float, default=1.0, help='m_prob')
parser.add_argument("--K_neigs", type=int, default=100, help='K_neigs')
parser.add_argument("--p_tri", type=float, default=0.8, help='sample probability for tri-neighbor set')
parser.add_argument("--p_bi", type=float, default=0.15, help='sample probability for bi-neighbor set')
parser.add_argument("--positive_neighbor_num", type=int, default=1000, help='num of node pairs in positive neighbors set')
parser.add_argument("--print_freq", type=int, default=5, help='print_freq')
parser.add_argument("--edge_type", type=str, default='euclid', help='euclid or pearson')
parser.add_argument("--is_probH", type=bool, default=False, help='prob edge True or False')
parser.add_argument("--use_rna", type=bool, default=1, help='use rna modality')
parser.add_argument("--use_adt", type=bool, default=1, help='use adt')
parser.add_argument("--use_atac", type=bool, default=1, help='use atac')
parser.add_argument("--n_hid", type=int, default=128, help='dimension of hidden layer')
parser.add_argument("--drop_out", type=float, default=0.1, help='dropout rate')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
# parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
parser.add_argument("--milestones", type=int, default=[100], help='milestones')
parser.add_argument("--gamma", type=float, default=0.9, help='gamma')
parser.add_argument("--weight_decay", type=float, default=0.0005, help='weight_decay')
parser.add_argument("--max_epoch", type=int, default=200, help='max_epoch')
parser.add_argument("--tau", type=float, default=1, help='temperature Coefficient')
parser.add_argument("--alpha", type=float, default=0.05, help='balanced factor for dual contrastive loss')
parser.add_argument("--beta", type=float, default=2., help='non-negative control parameter for intra_cell_loss')
parser.add_argument("--labeled_cell_ratio", type=float, default=0.02, help='labeled cell ratio for cell type annotation')
parser.add_argument('--alpha1', type=float, default=1.0, help='Weight for L_REC1')
parser.add_argument('--alpha2', type=float, default=1.0, help='Weight for L_REC2')
parser.add_argument('--lambda1', type=float, default=1.0, help='Weight for L_kl_loss1')
parser.add_argument('--lambda2', type=float, default=1.5, help='Weight for loss_contrastive_1')
parser.add_argument('--lambda3', type=float, default=1.5, help='Weight for loss_contrastive_2')
parser.add_argument('--lambda4', type=float, default=1.5, help='Weight for loss_contrastive_3')
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据准备
def load_and_prepare_data(args):
    # 加载数据
    fts, lbls, H, H_rna, H_atac = load_feature_and_H(
        args.data_dir_rna,
        args.data_dir_atac,
        args.lbls_dir,
        m_prob=args.m_prob,
        K_neigs=args.K_neigs,
        is_probH=args.is_probH,
        edge_type=args.edge_type,
        use_rna=args.use_rna,
        use_atac=args.use_atac
    )

    # 生成图结构和邻居数据
    G = generate_G_from_H(H)
    N = fts.shape[0]
    n_class = int(lbls.max()) + 1
    H_tri, H_bi, H_single, H_none = generate_node_pair_sets(H_rna, H_atac)

    must_include_idx = [sample_(i, lbls) for i in range(n_class)]

    # 数据转换到设备
    fts = torch.Tensor(fts).to(device)
    lbls = torch.Tensor(lbls).squeeze().long().to(device)
    G = torch.Tensor(G).to(device)
    H_none = torch.from_numpy(H_none).to(device)

    return fts, lbls, G, H_none, H_tri, H_bi, H_single, N

# 定义用于 embedding1 的 MLP 投影层
class MLPProjectionForEmbedding1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjectionForEmbedding1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # 使用 GELU 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.gelu(x)  # 使用 GELU 激活
        x = self.fc2(x)
        return x


# 定义用于 Z1 和 Z2 的 MLP 投影层
class MLPProjectionForZ(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjectionForZ, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # 使用 GELU 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.gelu(x)  # 使用 GELU 激活
        x = self.fc2(x)
        return x
# 定义对比学习损失
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        # 计算正样本和负样本
        positive_samples = torch.cat(
            (torch.diag(sim, self.batch_size).unsqueeze(1),
             torch.diag(sim, -self.batch_size).unsqueeze(1)), dim=0
        )
        negative_samples = sim[self.mask].reshape(N, -1)

        # 创建标签
        labels = torch.zeros(positive_samples.size(0)).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss


def train_combined_model(model1, model2, criterion_drr, criterion_rec, criterion_intra_cell, criterion_inter_cell, optimizer1, optimizer2, scheduler1, scheduler2, num_epochs=25, print_freq=500):
    loss_train = []
    epoch_all = []

    # 定义对比损失
    instance_loss = InstanceLoss(batch_size=fts.shape[0], temperature=args.tau).to(device)

    # 初始化降维MLP
    projection_layer = MLPProjectionForEmbedding1(input_dim=args.n_hid, hidden_dim=64, output_dim=20).to(device)

    for epoch in range(num_epochs):
        epoch_all.append(epoch)
        model1.train()
        model2.train()

        # 样本邻域采样
        coor_sampled_tri = neighbor_sampling(H_tri, args.positive_neighbor_num, args.p_tri)
        coor_sampled_bi = neighbor_sampling(H_bi, args.positive_neighbor_num, args.p_bi)
        coor_sampled_single = neighbor_sampling(H_single, args.positive_neighbor_num, 1 - args.p_tri - args.p_bi)
        H_union = neighbor_concat(coor_sampled_tri, coor_sampled_bi, coor_sampled_single, N)
        H_union = torch.from_numpy(H_union).to(device)

        x_ach, x_pos, x_neg = model1(fts, G)
        loss_intra_cell = criterion_intra_cell(x_ach, x_pos, x_neg)
        loss_inter_cell = criterion_inter_cell(x_ach, H_union, H_none)

        # 获取 model1 的输出 (embedding1)
        embedding1, _, _ = model1(fts, G)
        # 使用MLP对embedding1降维到20维
        embedding1_proj = projection_layer(embedding1)

        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, cons1, mu1, log_var1 = model2(X1, A1, X2, A2)

        concatenated = np.concatenate((embedding1_proj.detach().numpy(), Z1.detach().numpy(), Z2.detach().numpy()), axis=1)


        # 进行K-means聚类
        kmeans = KMeans(num_clusters, random_state=42)
        labels_pred = kmeans.fit_predict(concatenated)
        # 获取模型2输出（如果有需要）
        # X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, cons2 = model2(X1, A1, X2, A2)
        # 计算聚类性能指标
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        acc = cluster_acc(labels_true, labels_pred)
        ari = adjusted_rand_score(labels_true, labels_pred)
        ami = adjusted_mutual_info_score(labels_true, labels_pred)

        # 打印聚类性能
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}, NMI: {nmi:.4f}, ACC: {acc:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}")

        # 对比学习损失（计算embedding1与Z1、Z2之间的损失）
        loss_contrastive_1 = instance_loss(embedding1_proj, Z1)
        loss_contrastive_2 = instance_loss(embedding1_proj, Z2)
        loss_contrastive_3 = instance_loss(Z1, Z2)

        # 损失计算
        L_DRR = criterion_drr(cons1)
        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)

        # 计算KL损失
        kl_loss1 = kl_loss(mu1, log_var1)  # 使用你定义好的KL损失函数

        # 综合损失（可以根据需要加权）
        # loss = L_DRR + args.alpha1 * L_REC1 + args.alpha2 * L_REC2 + args.beta * loss_intra_cell + args.tau * loss_inter_cell
        loss = L_DRR + args.alpha1 * L_REC1 + args.alpha2 * L_REC2 + args.beta * loss_intra_cell + args.tau * loss_inter_cell + args.lambda1 * kl_loss1 + args.lambda2 * loss_contrastive_1 + args.lambda3 * loss_contrastive_2 + args.lambda4 * loss_contrastive_3

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        scheduler1.step()  # 确保在 optimizer1.step() 之后调用
        scheduler2.step()  # 确保在 optimizer2.step() 之后调用

        loss_train.append(loss.item())

        if epoch % print_freq == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}',
                  f'Loss: {loss.item():.4f}',
                  f'L_DRR: {L_DRR.item():.4f}',
                  f'L_REC1: {L_REC1.item():.4f}',
                  f'L_REC2: {L_REC2.item():.4f}',
                  f'Loss_intra_cell: {loss_intra_cell.item():.4f}',
                  f'Loss_inter_cell: {loss_inter_cell.item():.4f}'
                  )

        # 在训练结束后输出聚类的平均结果
        print("\nTraining finished! Calculating average clustering metrics...")
        avg_nmi = np.mean(nmi)
        avg_acc = np.mean(acc)
        avg_ari = np.mean(ari)
        avg_ami = np.mean(ami)

        print(f"Average NMI: {avg_nmi:.4f}")
        print(f"Average ACC: {avg_acc:.4f}")
        print(f"Average ARI: {avg_ari:.4f}")
        print(f"Average AMI: {avg_ami:.4f}")

    model1.eval()
    model2.eval()

    # 获取模型1的嵌入
    embedding1, _, _ = model1(fts, G)
    # 获取模型2的嵌入
    _, _, _, _, _, _, _, _, Z1, Z2, _, mu1, log_var1 = model2(X1, A1, X2, A2)
    # _, _, _, _, _, _, _, _, Z1, Z2, _ = model2(X1, A1, X2, A2)
    # 合并嵌入
    # concatenated_embedding = torch.cat([embedding1, Z1, Z2], dim=1)
    # 对embedding1进行降维
    embedding1_proj = projection_layer(embedding1)

    return model1, model2, loss_train, epoch_all, embedding1_proj, Z1, Z2, mu1, log_var1

def train_model_combined():
    print("Training combined HighOrder_model...")

    # 初始化模型
    model_ft = HGNN_unsupervised(in_ch=fts.shape[1], n_hid=args.n_hid, dropout=args.drop_out).to(device)
    model = scMIC(ae1, ae2, gae1, gae2, n_node=X1.shape[0]).to(device)  # 确保模型和设备匹配

    # 初始化优化器
    optimizer_ft = Adam(model_ft.parameters(), lr=opt.args.lr)
    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    # 初始化学习率调度器
    scheduler_ft = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=args.milestones, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # 定义损失函数
    criterion_drr = drr_loss  # 确保 drr_loss 是定义的损失函数
    criterion_rec = reconstruction_loss  # 确保 reconstruction_loss 是定义的损失函数
    criterion_intra_cell = intra_cell_loss(args.beta)
    criterion_inter_cell = inter_cell_loss(args.tau)


    model1, model2, loss_train, epoch_all, embedding1_proj, Z1, Z2, mu1, log_var1 = train_combined_model(
        model_ft, model, criterion_drr, criterion_rec, criterion_intra_cell, criterion_inter_cell,
        optimizer_ft, optimizer, scheduler_ft, scheduler,
        num_epochs=args.max_epoch,
        print_freq=args.print_freq
    )

    # 保存嵌入
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    np.save(os.path.join(output_dir, 'embedding1_proj.npy'), embedding1_proj.cpu().detach().numpy())
    np.save(os.path.join(output_dir, 'Z1.npy'), Z1.cpu().detach().numpy())
    np.save(os.path.join(output_dir, 'Z2.npy'), Z2.cpu().detach().numpy())
    # np.save(os.path.join(output_dir, 'combined_model_embedding.npy'), concatenated_embedding.cpu().detach().numpy())

if __name__ == '__main__':
    # 设置
    print("setting:")
    setup_seed(opt.args.seed)
    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("lambda1 value : {}".format(opt.args.lambda1))
    print("lambda2 value : {}".format(opt.args.lambda2))
    print("lambda3 value : {}".format(opt.args.lambda3))
    print("alpha1 value  : {:.0e}".format(args.alpha1))
    print("alpha2 value  : {:.0e}".format(args.alpha2))
    print("k value       : {}".format(opt.args.k))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

    # 加载数据
    fts, lbls, G, H_none, H_tri, H_bi, H_single, N = load_and_prepare_data(args)
    labels_path = "F:/1-scMIC2-main/scMIC-main/input/Ma-2020-4/lbls.npy"
    labels_true = np.load(labels_path)  # 加载标签
    num_clusters = len(np.unique(labels_true))
    X1, y, A1 = load_data(opt.args.name, 'rna', opt.args.method, opt.args.k, show_details=False)
    X2, y, A2 = load_data(opt.args.name, 'atac', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)

    X1 = torch.tensor(X1, dtype=torch.float32).to(device)
    A1 = torch.tensor(A1, dtype=torch.float32).to(device)
    X2 = torch.tensor(X2, dtype=torch.float32).to(device)
    A2 = torch.tensor(A2, dtype=torch.float32).to(device)

    # 自动获取输入特征维度
    n_d1 = X1.shape[1]  # X1的特征维度
    n_d2 = X2.shape[1]  # X2的特征维度

    # 初始化模型
    ae1 = VAE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=n_d1, n_z=opt.args.n_z).to(opt.args.device)

    ae2 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=n_d2, n_z=opt.args.n_z).to(opt.args.device)

    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)
    # 初始化和训练模型
    t0 = time.time()
    train_model_combined()
    t1 = time.time()
    print("Time_cost: {}".format(t1 - t0))