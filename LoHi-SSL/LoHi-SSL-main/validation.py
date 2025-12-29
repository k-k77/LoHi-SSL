from operator import itemgetter
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import silhouette_score,adjusted_rand_score,homogeneity_score,normalized_mutual_info_score,adjusted_mutual_info_score,calinski_harabasz_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score,accuracy_score
import umap
import warnings
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from operator import itemgetter
parser = argparse.ArgumentParser()  # 用于解析命令行参数

parser.add_argument("--dataset", type=str, default='sim1', help='used dataset name')
parser.add_argument("--supervised", type=bool, default=0, help='True for stage 2 validation and False for stage1')
args = parser.parse_args()
warnings.filterwarnings("ignore")


if args.dataset == 'sim1':
    label_dict = {'6_1':0,'6_2':1,'6_3':2,'6_4':3,'6_5':4}
elif args.dataset == 'sim2':
    label_dict = {'9_1':0,'9_2':1,'9_3':2,'9_4':3,'9_5':4,'9_6':5,'9_7':6,'9_8':7}
elif args.dataset == 'sim3':
    label_dict = {'13_1':0,'13_2':1,'13_3':2,'13_4':3,'13_5':4,'13_6':5,'13_7':6,'13_8':7,'13_9':8,'13_10':9,'13_11':10,'13_12':11}


def purity_score(y_true, y_pred):  # 计算聚类结果的纯度分数
    contingency_matrix1 = contingency_matrix(y_true, y_pred)  # 计算真实类别标签与预测类别标签之间的列联表
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 
# np.amax(contingency_matrix1, axis=0)：在列联表的每一列中找到最大值，这表示每个预测类别中真实类别标签的最多出现次数
# np.sum(np.amax(contingency_matrix1, axis=0))：计算所有预测类别中真实类别标签的最多出现次数的总和
label_dict_ = {}
for key, value in label_dict.items():
    label_dict_[value] = key
    






