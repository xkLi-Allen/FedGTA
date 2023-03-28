import torch
import platform
import numpy as np
import torch.nn as nn
import scipy.sparse as sp

from torch import Tensor
from operators.utils import csr_sparse_dense_matmul, cuda_csr_sparse_dense_matmul


class SimGraphOp:
    def __init__(self, prop_steps):
        self.prop_steps = prop_steps
        self.adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj = self.construct_adj(adj)
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            if isinstance(feature, Tensor):
                feature = feature.numpy()
            else:
                raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
        prop_feat_list = [feature]
        for _ in range(self.prop_steps):
            feat_temp = ada_platform_one_step_propagation(self.adj, prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
        return [torch.FloatTensor(feat) for feat in prop_feat_list]


class SimMessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(SimMessageOp, self).__init__()
        self.aggr_type = None
        self.start, self.end = start, end

    def aggr_type(self):
        return self.aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")
        return self.combine(feat_list)
    
def ada_platform_one_step_propagation(adj, x):
    if platform.system() == "Linux":
        one_step_prop_x = csr_sparse_dense_matmul(adj, x)
    else:
        one_step_prop_x = adj.dot(x)
    return one_step_prop_x