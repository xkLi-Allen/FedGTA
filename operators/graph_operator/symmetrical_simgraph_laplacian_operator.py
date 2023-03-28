import scipy.sparse as sp

from operators.base_operator import SimGraphOp
from operators.utils import adj_to_symmetric_norm


class SymSimLaplacianGraphOp(SimGraphOp):
    def __init__(self, prop_steps, r=0.5):
        super(SymSimLaplacianGraphOp, self).__init__(prop_steps)
        self.r = r

    def construct_adj(self, adj):
        adj = adj.tocoo()
        adj_normalized = adj_to_symmetric_norm(adj, self.r)
        return adj_normalized.tocsr()
