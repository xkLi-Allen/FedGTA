import torch
import torch.nn as nn


class SimMultiLayerPerceptron(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, output_dim, dropout, bn=False):
        super(SimMultiLayerPerceptron, self).__init__()
        self.adj = None
        self.query_edges = None
        if num_layers < 2:
            raise ValueError("MLP must have at least two layers!")
        self.num_layers = num_layers
        self.linear = nn.Linear(2 * hidden_dim, output_dim)
        self.fcs_node_edge = nn.ModuleList()
        self.fcs_node_edge.append(nn.Linear(feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.fcs_node_edge.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcs_node_edge.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        if self.bn is True:
            self.bns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs_node_edge:
            nn.init.xavier_uniform_(fc.weight, gain=gain)
            nn.init.zeros_(fc.bias)

    def forward(self, feature):
        for i in range(self.num_layers - 1):
            feature = self.fcs_node_edge[i](feature)
            if self.bn is True:
                feature = self.bns[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        if self.query_edges == None:
            output = self.fcs_node_edge[-1](feature)
        else:
            x = torch.cat((feature[self.query_edges[:, 0]], feature[self.query_edges[:, 1]]), dim=-1)
            x = self.dropout(x)
            output = self.linear(x)

        return output