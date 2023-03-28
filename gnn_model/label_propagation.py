import copy
import torch
import random
import platform
import torch.nn.functional as F

from data.utils import idx_to_mask
from utils import csr_sparse_dense_matmul
from utils import homo_adj_to_symmetric_norm


class NonParaLP():
    def __init__(self, prop_steps, num_class, subgraph, device, alpha=0.5, r=0.5, temperature=20):
        self.prop_steps = prop_steps
        self.r = r
        self.num_class = num_class
        self.alpha = alpha
        self.subgraph = subgraph.to(device)
        self.y = subgraph.y.to(device)
        self.device = device
        num_nodes = len(self.subgraph.train_idx)
        train_idx_list = torch.where(self.subgraph.train_idx == True)[0].cpu().numpy().tolist()
        random.shuffle(train_idx_list)
        self.label_idx = idx_to_mask(train_idx_list, num_nodes).to(device)
        self.unlabel_idx = self.subgraph.val_idx | self.subgraph.test_idx
        self.temperature = temperature
        self.label = F.one_hot(self.y.view(-1), self.num_class).to(torch.float).to(self.device)  
        self.adj = homo_adj_to_symmetric_norm(self.subgraph.adj, r=r)

    def preprocess(self, soft_label):
        unlabel_init = soft_label[self.unlabel_idx].to(self.device)
        self.label[self.unlabel_idx] = unlabel_init

    def eval(self):
        pred = self.output_2.max(1)[1].type_as(self.subgraph.y)
        tot_correct = pred.eq(self.subgraph.y).double()
        tot_correct = tot_correct.sum()
        tot_reliability_acc = (tot_correct / self.subgraph.y.shape[0]).item()
        correct = pred[self.label_idx].eq(self.subgraph.y[self.label_idx]).double()
        correct = correct.sum()
        reliability_acc = (correct / self.subgraph.y[self.label_idx].shape[0]).item()
        return tot_reliability_acc, reliability_acc

    def init_lp_propagate(self, feature, init_label, alpha):
        init_label_ = copy.deepcopy(init_label.cpu())
        feature = feature.cpu().numpy()
        feat_temp = feature
        for _ in range(self.prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self.adj, feat_temp)
            else:
                feat_temp = self.adj.dot(feat_temp)

            feat_temp = alpha * feat_temp + (1 - alpha) * feature
            feat_temp[init_label_] += feature[init_label_]
        return torch.tensor(feat_temp)

    def propagate(self):
        self.output = self.init_lp_propagate(self.label, init_label=self.label_idx, alpha=self.alpha).to(self.device)
        self.output_raw = F.softmax(self.output, dim=1)
        self.output_dis = F.softmax(self.output/self.temperature, dim=1)
        self.output_raw[self.label_idx] = self.label[self.label_idx]
        self.output_dis[self.label_idx] = self.label[self.label_idx]
        return self.output_raw, self.output_dis
