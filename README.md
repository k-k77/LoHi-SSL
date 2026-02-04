Markdown# LoHi-SSL: Multi-Level Synergistic Learning for Single-Cell Multi-Omics Integration

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)

## Overview

**LoHi-SSL** is a deep learning framework designed for the effective integration of single-cell multi-omics data (e.g., scRNA-seq and scATAC-seq). It addresses the challenges of cross-modal heterogeneity and complex cellular interactions by synergizing **Low-order information** (via Autoencoders and Graph Autoencoders) and **High-order information** (via Hypergraph Neural Networks).

Key features:
- **Low-Order Fusion:** Captures local neighborhood structures and intrinsic cellular attributes.
- **High-Order Fusion:** Models complex, non-pairwise cellular correlations using hypergraphs.
- **Contrastive Learning:** Aligns features across modalities and levels to enhance discriminability.

## Directory Structure

Please ensure your project directory is organized as follows:

```text
LoHi-SSL/
├── input/                  # Data directory
├── output/                 # Model outputs (embeddings, logs)
├── HighOrder_model/        # Hypergraph model modules
│   ├── H_model.py
│   └── loss.py
├── contrastive_loss.py
├── data_loader.py
├── encoder.py
├── layer.py
├── LowOrder.py
├── main.py                 # Entry point for training
├── multiomics_hypergraph_construction.py
├── opt.py                  # Configuration and parameters
├── utils.py
├── validation.py
└── requirements.txt        # Dependency list
Environment & InstallationThe code has been tested with Python 3.9. To ensure reproducibility, we recommend creating a dedicated virtual environment using Conda.1. Clone the repositoryBashgit clone [https://github.com/k-k77/LoHi-SSL.git](https://github.com/k-k77/LoHi-SSL.git)
cd LoHi-SSL
2. Create a virtual environmentBashconda create -n lohi_ssl python=3.9
conda activate lohi_ssl
3. Install dependenciesInstall the required Python packages using the provided requirements.txt:Bashpip install -r requirements.txt
Core DependenciesThe specific versions used in this research are listed below for reference:PackageVersiontorch2.8.0numpy1.25.2pandas2.3.3scipy1.13.1scikit-learn1.6.1scanpy1.9.8anndata0.9.2matplotlib3.8.4umap-learn0.5.1munkres1.1.4> Note: For GPU acceleration, please ensure your PyTorch installation is compatible with your system's CUDA version.UsageTo train the model and evaluate clustering performance, run main.py. The script expects the data to be placed in the ./input/ directory.Basic CommandBashpython main.py --dataset Ma-2020-4 --data_dir_rna ./input/Ma-2020-4/rna.npy --data_dir_atac ./input/Ma-2020-4/atac.npy --lbls_dir ./input/Ma-2020-4/lbls.npy
Key ArgumentsYou can adjust hyperparameters via command-line arguments. See main.py and opt.py for a full list.ArgumentDefaultDescription--datasetsim1Name of the dataset to use.--max_epoch200Number of training epochs.--lr0.001Learning rate for the optimizer.--n_hid128Dimension of the hidden layer.--k20Number of nearest neighbors for graph construction.--K_neigs100Number of neighbors for hypergraph construction.--tau1.0Temperature coefficient for contrastive loss.--lambda1...4VariesWeights for different loss components (KL, contrastive, etc.).Example: Running with Custom HyperparametersBashpython main.py --dataset Ma-2020-4 --lr 0.0001 --max_epoch 300 --n_hid 64 --tau 0.5
CitationIf you use this code or model in your research, please cite our paper:代码段@article{LoHi-SSL2024,
  title={LoHi-SSL: A Multi-Level Synergistic Learning Model for Integrating Single-Cell Multi-Omics Data via Low- and High-Order Information Fusion},
  author={Xiaoyun Xiong, Kaihao Zhang, Chengdong Zhang, Yuanyuan Zhang},
  journal={Journal Name},
  year={2024}
}
ContactFor any questions or issues, please open an issue in this repository or contact the authors.
