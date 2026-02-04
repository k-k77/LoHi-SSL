Markdown# LoHi-SSL: Multi-Level Synergistic Learning for Single-Cell Multi-Omics Integration

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)

---

## 📖 Overview

**LoHi-SSL** is a deep learning framework designed for the effective integration of single-cell multi-omics data (e.g., scRNA-seq and scATAC-seq). It addresses the challenges of cross-modal heterogeneity and complex cellular interactions by synergizing:

* **Low-order information:** Via Autoencoders and Graph Autoencoders.
* **High-order information:** Via Hypergraph Neural Networks.

---

## 📂 Directory Structure

Please ensure your project directory is organized as follows:

```text
LoHi-SSL/
├── input/                  # Place your dataset files (.npy) here
├── HighOrder_model/        # Hypergraph model modules
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
⚙️ Environment & DependenciesThe code has been tested with Python 3.9. To ensure reproducibility, we recommend creating a dedicated virtual environment.1. Setup EnvironmentBashgit clone [https://github.com/k-k77/LoHi-SSL.git](https://github.com/k-k77/LoHi-SSL.git)
cd LoHi-SSL
conda create -n lohi_ssl python=3.9
conda activate lohi_ssl
2. Install DependenciesYou can install all required packages using the requirements.txt file provided in this repository:Bashpip install -r requirements.txt
📋 Specific VersionsFor full reproducibility, the specific package versions used in this research are listed below:
PackageVersiontorch2.8.0
numpy1.25.2
pandas2.3.3
scipy1.13.1
scikit-learn1.6.1
scanpy1.9.8
anndata0.9.2
matplotlib3.8.4
umap-learn0.5.1
munkres1.1.4
tqdm>=4.50.0
Note: For GPU acceleration, please ensure your PyTorch installation is compatible with your system's CUDA version.
🚀 UsageTo train the model and evaluate clustering performance, run main.py.
Key ArgumentsYou can adjust hyperparameters via command-line arguments. See main.py and opt.py for a full list.
ArgumentDefaultDescription--datasetsim1Name of the dataset to use.
--max_epoch200Number of training epochs.
--lr0.001Learning rate.
--n_hid128Dimension of the hidden layer.
--tau1.0Temperature coefficient for contrastive loss.Example with Custom ParametersBashpython main.py
--dataset Ma-2020-4
--lr 0.0001
--max_epoch 300
--n_hid 64
📝 CitationIf you use this code or model in your research, please cite our paper:代码段@article{LoHi-SSL2024,
  title={LoHi-SSL: A Multi-Level Synergistic Learning Model for Integrating Single-Cell Multi-Omics Data via Low- and High-Order Information Fusion},
  author={Xiaoyun Xiong, Kaihao Zhang, Chengdong Zhang, Yuanyuan Zhang},
  journal={Journal Name},
  year={2024}
}
📧 ContactFor any questions or issues, please open an issue in this repository or contact the authors.
