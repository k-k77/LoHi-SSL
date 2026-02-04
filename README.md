Markdown# LoHi-SSL: Multi-Level Synergistic Learning for Single-Cell Multi-Omics Integration

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg)](https://pytorch.org/)

---

## 📖 Overview

[cite_start]**LoHi-SSL** is a deep learning framework designed for the effective integration of single-cell multi-omics data (e.g., scRNA-seq and scATAC-seq)[cite: 1, 3, 5]. It addresses the challenges of cross-modal heterogeneity and complex cellular interactions by synergizing:

* [cite_start]**Low-order information:** Via Autoencoders and Graph Autoencoders[cite: 5, 6].
* [cite_start]**High-order information:** Via Hypergraph Neural Networks[cite: 5, 6].

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
⚙️ Environment & InstallationThe code has been tested with Python 3.9. To ensure reproducibility, we recommend creating a dedicated virtual environment.1. Clone the repositoryBashgit clone [https://github.com/k-k77/LoHi-SSL.git](https://github.com/k-k77/LoHi-SSL.git)
cd LoHi-SSL
2. Create and activate environmentBashconda create -n lohi_ssl python=3.9
conda activate lohi_ssl
3. Install dependenciesBashpip install -r requirements.txt
Core Dependencies (Reference)The specific versions used in this research are:torch==2.8.0numpy==1.25.2scanpy==1.9.8scikit-learn==1.6.1matplotlib==3.8.4Note: For GPU acceleration, please ensure your PyTorch installation is compatible with your system's CUDA version.
🚀 UsageTo train the model and evaluate clustering performance, run main.py.
Key ArgumentsYou can adjust hyperparameters via command-line arguments. See main.py and opt.py for a full list.ArgumentDefaultDescription--datasetsim1Name of the dataset to use.--max_epoch200Number of training epochs.--lr0.001Learning rate.--n_hid128Dimension of the hidden layer.--tau1.0Temperature coefficient for contrastive loss.Example with Custom ParametersBashpython main.py --dataset Ma-2020-4 --lr 0.0001 --max_epoch 300 --n_hid 64
📝 CitationIf you use this code or model in your research, please cite our paper:代码段@article{LoHi-SSL2024,
  title={LoHi-SSL: A Multi-Level Synergistic Learning Model for Integrating Single-Cell Multi-Omics Data via Low- and High-Order Information Fusion},
  author={Xiaoyun Xiong, Kaihao Zhang, Chengdong Zhang, Yuanyuan Zhang},
  journal={Journal Name},
  year={2024}
}
📧 ContactFor any questions or issues, please open an issue in this repository or contact the authors.
