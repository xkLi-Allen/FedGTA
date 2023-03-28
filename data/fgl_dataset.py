import os
import time
import torch
import os.path as osp
import scipy.sparse as sp

from torch import Tensor
from scipy.sparse import csr_matrix
from data.utils import data_partition
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


class FGLDataset(Dataset):
    def __init__(self, root, name, num_clients, partition, train=0.2, val=0.4, test=0.4, view_subgraph=False, device=torch.device("cpu"), transform=None, pre_transform=None, pre_filter=None):
        start = time.time()
        self.root = root
        self.name = name
        self.num_clients = num_clients
        self.partition = partition
        self.view_subgraph = view_subgraph
        self.train = train
        self.val = val
        self.test = test
        super(FGLDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_data()
        for i in range(num_clients):
            self.subgraphs[i].y = self.subgraphs[i].y.to(device)
        end = time.time()
        print(f"load FGL dataset {name} done ({end-start:.2f} sec)")
        self.num_classes = self.global_dataset.num_classes

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return osp.join(self.raw_dir, self.name, "Client{}".format(self.num_clients), self.partition)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ['data{}.pt'.format(i) for i in range(self.num_clients)]
        return files_names

    def download(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data{}.pt'.format(idx)))
        return data

    def process(self):
        self.load_global_graph()
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        subgraph_list = data_partition(
            G=self.global_data,
            num_clients=self.num_clients,
            train=self.train,
            val=self.val,
            test=self.test,
            partition=self.partition
        )
        for i in range(self.num_clients):
            torch.save(subgraph_list[i], self.processed_paths[i])

    def load_global_graph(self):
        self.global_dataset = Planetoid(root=self.raw_dir, name=self.name)
        self.input_dim = self.global_dataset.num_features
        self.output_dim = self.global_dataset.num_classes
        self.global_data = self.global_dataset.data
        self.global_data.adj = sp.coo_matrix((torch.ones([len(self.global_data.edge_index[0])]),
                                              (self.global_data.edge_index[0], self.global_data.edge_index[1])),
                                             shape=(self.global_data.num_nodes, self.global_data.num_nodes))
        self.global_data.row, self.global_data.col, self.global_data.edge_weight = self.global_data.adj.row, self.global_data.adj.col, self.global_data.adj.data
        if isinstance(self.global_data.row, Tensor) or isinstance(self.global_data.col, Tensor):
            self.global_data.adj = csr_matrix((self.global_data.edge_weight.numpy(),
                                               (self.global_data.row.numpy(), self.global_data.col.numpy())),
                                              shape=(self.global_data.num_nodes, self.global_data.num_nodes))
        else:
            self.global_data.adj = csr_matrix(
                (self.global_data.edge_weight, (self.global_data.row, self.global_data.col)),
                shape=(self.global_data.num_nodes, self.global_data.num_nodes))

    def load_data(self):
        self.load_global_graph()
        self.subgraphs = [self.get(i) for i in range(self.num_clients)]
        for i in range(len(self.subgraphs)):
            if i == 0:
                self.global_data.train_idx = self.subgraphs[i].global_train_idx
                self.global_data.val_idx = self.subgraphs[i].global_val_idx
                self.global_data.test_idx = self.subgraphs[i].global_test_idx
            else:
                self.global_data.train_idx += self.subgraphs[i].global_train_idx
                self.global_data.val_idx += self.subgraphs[i].global_val_idx
                self.global_data.test_idx += self.subgraphs[i].global_test_idx






