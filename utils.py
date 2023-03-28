import torch
import random
import numpy as np
import os.path as osp
import scipy.sparse as sp
import numpy.ctypeslib as ctl

from ctypes import c_int
from scipy.sparse import coo_matrix
from torchviz import make_dot


def view_grad(input, model_list=None):
    if model_list is not None:
        make_dot(input, dict([model.named_parameters() for model in model_list])).view()
    else:
        make_dot(input).view()

def sparseTensor_to_coomatrix(edge_idx, num_nodes):
    if edge_idx.shape == torch.Size([0]):
        adj = coo_matrix((num_nodes, num_nodes), dtype=np.int)
    else:
        row = edge_idx[0].cpu().numpy()
        col = edge_idx[1].cpu().numpy()
        data = np.ones(edge_idx.shape[1])
        adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.int)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    
def homo_adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized

def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    ctl_lib = ctl.load_library("./models/csrc/libmatmul.so", dir_path)
    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None
    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape
    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)
    return answer.reshape(feature.shape)

def cuda_csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]
    
    ctl_lib = ctl.load_library("./models/csrc/libcudamatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDense.argtypes = [arr_1d_float, c_int, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float, c_int,
                                         c_int]
    ctl_lib.FloatCSRMulDense.restypes = c_int

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    data_nnz = len(data)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDense(answer, data_nnz, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False