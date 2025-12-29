import argparse
import torch

parser = argparse.ArgumentParser(description='LowOrder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ==========================================
# 1. 数据集与路径设置 
# ==========================================
parser.add_argument('--name', type=str, default="Ma-2020-1", help="Dataset name")
parser.add_argument("--dataset", type=str, default='Ma-2020-1', help='Dataset name (alias for name)')
parser.add_argument('--data_dir_rna', default='./input/Ma-2020-1/rna.npy', help='path of RNA data')
parser.add_argument('--data_dir_atac', default='./input/Ma-2020-1/atac.npy', help='path of ATAC data')
parser.add_argument('--lbls_dir', default='./input/Ma-2020-1/lbls.npy', help='path of cell lbls')

# ==========================================
# 2. 训练控制参数
# ==========================================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='Gpu')

# Epoch 设置
parser.add_argument('--rec_epoch', type=int, default=30)
parser.add_argument('--fus_epoch', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=100, help='Used in main loop')
parser.add_argument('--epoch', type=int, default=500) 

parser.add_argument('--print_freq', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--milestones', type=int, default=[100])
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--pretrain', type=bool, default=False)

# ==========================================
# 3. 损失权重参数 (Loss Weights)
# ==========================================
# 对比学习权重
parser.add_argument('--lambda1', type=float, default=2.0) 
parser.add_argument('--lambda2', type=float, default=0.1) 
parser.add_argument('--lambda3', type=float, default=2.0) 
parser.add_argument('--lambda4', type=float, default=1.5)

# 重构与结构权重
parser.add_argument('--alpha1', type=float, default=1.0) # REC1
parser.add_argument('--alpha2', type=float, default=1.0) # REC2
parser.add_argument('--alpha_value', type=float, default=0.1) # Balance factor in rec loss
parser.add_argument('--beta', type=float, default=2.0)   # intra_cell_loss
parser.add_argument('--tau', type=float, default=1.0)    # temperature

# ==========================================
# 4. 模型结构参数 (Model Architecture)
# ==========================================
# 维度设置
parser.add_argument('--n_d1', type=int, default=100)
parser.add_argument('--n_d2', type=int, default=100)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--n_hid', type=int, default=128) # HighOrder hidden dim
parser.add_argument('--dropout', type=float, default=***)

# AE structure
parser.add_argument('--ae_n_enc_1', type=int, default=256)
parser.add_argument('--ae_n_enc_2', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=256)

# IGAE structure
parser.add_argument('--gae_n_enc_1', type=int, default=256)
parser.add_argument('--gae_n_enc_2', type=int, default=128)
parser.add_argument('--gae_n_dec_1', type=int, default=128)
parser.add_argument('--gae_n_dec_2', type=int, default=256)

# ==========================================
# 5. 图构建与 HighOrder 参数 (Graph & Sampling)
# ==========================================
parser.add_argument('--K_neigs', type=int, default=100)
parser.add_argument('--positive_neighbor_num', type=int, default=1000)
parser.add_argument('--p_tri', type=float, default=0.8)
parser.add_argument('--p_bi', type=float, default=0.15)
parser.add_argument('--edge_type', type=str, default='euclid')
parser.add_argument('--is_probH', type=bool, default=False)
parser.add_argument('--use_rna', type=bool, default=True)
parser.add_argument('--use_atac', type=bool, default=True)

# ==========================================
# 6. LowOrder 数据加载参数
# ==========================================
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--method', type=str, default='euc')
parser.add_argument('--first_view', type=str, default='ATAC')

# ==========================================
# 7. 运行时动态参数 (Runtime Placeholders)
# ==========================================

parser.add_argument('--n_clusters', type=int, default=0)

# Performance metrics placeholders (optional)
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)

args = parser.parse_args()

