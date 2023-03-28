import torch
import platform
import numpy as np
import scipy.sparse as sp

from utils import csr_sparse_dense_matmul


def adj_to_symmetric_norm(adj, r,decay=0.8): # decay < 1;  if decay > 1 <=> no decay
    if isinstance(adj, sp.csr_matrix):
        adj = adj.tocoo()
    elif not isinstance(adj, sp.coo_matrix):
        raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix.maximum(decay * adj, sp.eye(adj.shape[0]))
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized.tocsr()



def propagate(prop_steps, adj, feature, r, decay=None):
    adj_norm = adj_to_symmetric_norm(adj, r)
    if not isinstance(feature, np.ndarray):
        feature = feature.numpy()
    if not isinstance(adj_norm, sp.csr_matrix):
        raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
    elif not isinstance(feature, np.ndarray):
        raise TypeError("The feature matrix must be a numpy.ndarray!")
    elif adj_norm.shape[1] != feature.shape[0]:
        raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
    processed_features = [feature]
    for _ in range(prop_steps):
        if platform.system() == "Linux":
            processed_features.append( csr_sparse_dense_matmul(adj_norm, processed_features[-1]))
        else:
            processed_features.append(adj_norm.dot(processed_features[-1]))
    processed_features = [torch.FloatTensor(processed_feature) for processed_feature in processed_features]
    return processed_features[1:]


def decay_adj_to_symmetric_norm(adj, r, decay):
    if isinstance(adj, sp.csr_matrix):
        adj = adj.tocoo()
    elif not isinstance(adj, sp.coo_matrix):
        raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix.maximum(decay * adj, sp.eye(adj.shape[0]))
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized.tocsr()

def decay_propagate(prop_steps, adj, feature, r, decay):
    adj_norm_decay = decay_adj_to_symmetric_norm(adj, r, decay=decay)
    if not isinstance(feature, np.ndarray):
        feature = feature.numpy()
    if not isinstance(adj_norm_decay, sp.csr_matrix):
        raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
    elif not isinstance(feature, np.ndarray):
        raise TypeError("The feature matrix must be a numpy.ndarray!")
    elif adj_norm_decay.shape[1] != feature.shape[0]:
        raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")
    processed_feature = feature
    for _ in range(prop_steps):
        if platform.system() == "Linux":
            processed_feature = csr_sparse_dense_matmul(adj_norm_decay, processed_feature)
        else:
            processed_feature = adj_norm_decay.dot(processed_feature)
    processed_feature = torch.FloatTensor(processed_feature)
    return processed_feature

def origin_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.pow(x, moment)
    return torch.mean(tmp, dim=dim)

def mean_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.mean(x, dim=dim)
    if dim == 0:
        tmp = x - tmp.view(1, -1)
    else:
        tmp = x - tmp.view(-1,1)
    tmp = torch.pow(tmp, moment)
    return  torch.mean(tmp, dim=dim)

def GM_process_h(adj, feature, neighbor_fields=5,  num_moments=5, r=0.5, decay=0.8):
    Xagg_list = propagate(neighbor_fields, adj, feature, r, decay)
    moment_list = []
    for i in range(len(Xagg_list)):
        moment_list.append(compute_moment(Xagg_list[i], num_moments=num_moments, dim="h"))
    moment_tensor = torch.cat(moment_list)
    return moment_tensor

def GM_process_v(adj, feature, neighbor_fields=5,  num_moments=5, r=0.5, decay=0.8):
    Xagg_list = propagate(neighbor_fields, adj, feature, r, decay)
    moment_list = []
    for i in range(len(Xagg_list)):
        moment_list.append(compute_moment(Xagg_list[i], num_moments=num_moments, dim="v"))
    moment_tensor = torch.cat(moment_list)
    return moment_tensor

def compute_moment(x, num_moments=5, dim="h", moment_type="origin"):
    if moment_type == "origin":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = origin_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "mean":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = mean_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "hybrid":
        o_ = compute_moment(x, num_moments, dim, moment_type="origin")
        m_ = compute_moment(x, num_moments, dim, moment_type="mean")
        return torch.cat((o_, m_))
