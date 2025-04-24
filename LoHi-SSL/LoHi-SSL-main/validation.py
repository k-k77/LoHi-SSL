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
# 从命令行中指定数据集名称和验证方式，并根据不同的数据集名称自动设置对应的类别标签映射
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
    



if not args.supervised:  # 执行无监督的细胞聚类验证

    print('cell clustering validation')
    Y_label = np.load('./datasets/example_dataset_simulation1/lbls.npy')
    x_ach = np.load('./output/scMHNN_embedding.npy')
    # print(Y_label.shape)
    # print(x_ach.shape)

    ASW_learned = []
    PS_learned = []
    HS_learned = []

    embedding_name = []
    cell_name = []
    for i in range(x_ach.shape[1]):
        embedding_name.append(str(i))

    for i in range(x_ach.shape[0]):
        cell_name.append(str(i))
    embedding_name = pd.DataFrame(index=embedding_name)
    cell_name = pd.DataFrame(index=cell_name)
    adata_learned=ad.AnnData(x_ach,obs=cell_name,var=embedding_name)
    Y_label_list = itemgetter(*list(Y_label))(label_dict_)
    adata_learned.obs['cell_type'] = Y_label_list
    sc.pp.neighbors(adata_learned,use_rep='X')
    sc.tl.umap(adata_learned,n_components=2)
    sc.pl.umap(adata_learned, color="cell_type")
    plt.title('scMHNN')
    plt.savefig('./output/scMHNN_umap_{}.jpg'.format(args.dataset), bbox_inches='tight', dpi=800)

    for i in list(np.round(np.linspace(0.1,1.0,10),1)):
        print('resolution:',i)

        sc.tl.louvain(adata_learned,resolution = i,key_added = "louvain")  # best
        y_predict = adata_learned.obs['louvain']

        ASW_learned.append(np.round(silhouette_score(x_ach,y_predict),3))
        PS_learned.append(np.round(purity_score(Y_label,y_predict),3))
        HS_learned.append(np.round(homogeneity_score(Y_label,y_predict),3))

    print('cell clustering results:')
    print('ASW_learned = ',ASW_learned)
    print('PS_learned = ',PS_learned)
    print('HS_learned = ',HS_learned)

else:
    print('cell annotation validation')
    df_result = pd.DataFrame()
    lbls_test = np.load('./output/lbls_test.npy') 
    preds_test = np.load('./output/preds_test.npy')

    df_result['acc'] = [np.round(accuracy_score(lbls_test,preds_test),3)]
    df_result['f1w'] = [np.round(f1_score(lbls_test,preds_test, average='weighted'),3)]
    print(df_result)


