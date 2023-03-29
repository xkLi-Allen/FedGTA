## FedGTA: Topology-aware Averaging for Federated Graph Learning

**Requirements**

Hardware environment: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz, NVIDIA GeForce RTX 3090 with 24GB memory.

Software environment: Ubuntu 18.04.6, Python 3.9, PyTorch 1.11.0 and CUDA 11.8.

1. Please refer to [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the environments;
2. Run 'pip install -r requirements.txt' to download required packages;

**Training**

To train the model(s) in the paper

1. Please unzip Cora.zip/CiteSeer.zip/PubMed.zip to the current file directory location
2. Open main.py to train GAMLP (the best local scalable GNN model) with our proposed federated graph model optimization strategies (FedGTA).

    We provide Cora/CiteSeer/PubMed dataset under Louvain 10 clients split as example (hyperparameters in config.py).

    Run this command:

```python
  python main.py
```
