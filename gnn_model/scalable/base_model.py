import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from utils import sparse_mx_to_torch_sparse_tensor as scipy_sparse_mat_to_torch_sparse_tensor


class SimBaseSGModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, output_dim):
        super(SimBaseSGModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.naive_graph_op = None
        self.pre_graph_op, self.pre_msg_op = None, None
        self.post_graph_op, self.post_msg_op = None, None
        self.base_model = None
        self.processed_feat_list = None
        self.processed_feature = None
        self.pre_msg_learnable = False

    def preprocess(self, adj, feature):
        if self.pre_graph_op is not None:
            self.processed_feat_list = self.pre_graph_op.propagate(
                adj, feature)
            if self.pre_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                self.pre_msg_learnable = True
            else:
                self.pre_msg_learnable = False
                self.processed_feature = self.pre_msg_op.aggregate(
                    self.processed_feat_list)
        else:
            if self.naive_graph_op is not None:
                self.base_model.adj = self.naive_graph_op.construct_adj(adj)
                if not isinstance(self.base_model.adj, sp.csr_matrix):
                    raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
                elif self.base_model.adj.shape[1] != feature.shape[0]:
                    raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
                self.base_model.adj = scipy_sparse_mat_to_torch_sparse_tensor(self.base_model.adj)
            self.pre_msg_learnable = False
            self.processed_feature = torch.FloatTensor(feature)

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            if self.post_msg_op.aggr_type in [
                "proj_concat", "learnable_weighted", "iterate_learnable_weighted"]:
                raise ValueError(
                    "Learnable weighted message operator is not supported in the post-processing phase!")
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output

    def model_forward(self, idx, device, ori=None):
        return self.forward(idx, device, ori)

    def get_emb(self, idx, device):
        return self.forward(idx, device)
    
    def forward(self, idx, device, ori=None):
        processed_feature = None
        if self.base_model.adj != None:
            self.base_model.adj = self.base_model.adj.to(device)
            processed_feature = self.processed_feature.to(device)
            if ori is not None: self.base_model.query_edges = ori
        else:
            if idx is None and self.processed_feature is not None: idx = torch.arange(self.processed_feature.shape[0])
            if self.pre_msg_learnable is False:
                processed_feature = self.processed_feature[idx].to(device)
            else:
                transferred_feat_list = [feat[idx].to(
                    device) for feat in self.processed_feat_list]
                processed_feature = self.pre_msg_op.aggregate(
                    transferred_feat_list)
        output = self.base_model(processed_feature)
        return output[idx] if self.base_model.query_edges is None and self.base_model.adj != None else output
