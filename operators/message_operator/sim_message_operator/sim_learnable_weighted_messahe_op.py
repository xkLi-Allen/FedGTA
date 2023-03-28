import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter, Linear
from operators.base_operator import SimMessageOp
from operators.utils import one_dim_weighted_add, two_dim_weighted_add, squeeze_first_dimension


class SimLearnableWeightedMessageOp(SimMessageOp):
    def __init__(self, start, end, combination_type, *args):
        super(SimLearnableWeightedMessageOp, self).__init__(start, end)
        self.aggr_type = "learnable_weighted"
        if combination_type not in ["simple", "simple_allow_neg", "gate", "ori_ref", "jk"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'simple', 'simple_allow_neg', 'gate', 'ori_ref' or 'jk'.")
        self.combination_type = combination_type
        self.learnable_weight = None
        if combination_type == "simple" or combination_type == "simple_allow_neg":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the simple learnable weighted aggregator!")
            prop_steps = args[0]
            tmp_2d_tensor = torch.FloatTensor(1, prop_steps + 1)
            nn.init.xavier_normal_(tmp_2d_tensor)
            self.learnable_weight = Parameter(tmp_2d_tensor.view(-1))
        elif combination_type == "gate":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the gate learnable weighted aggregator!")
            feat_dim = args[0]
            self.learnable_weight = Linear(feat_dim, 1)
        elif combination_type == "ori_ref":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the ori_ref learnable weighted aggregator!")
            feat_dim = args[0]
            self.learnable_weight = Linear(feat_dim + feat_dim, 1)
        elif combination_type == "jk":
            if len(args) != 2:
                raise ValueError(
                    "Invalid parameter numbers for the jk learnable weighted aggregator!")
            prop_steps, feat_dim = args[0], args[1]
            self.learnable_weight = Linear(
                feat_dim + (prop_steps + 1) * feat_dim, 1)
    def combine(self, feat_list):
        weight_list = None
        feat_list = squeeze_first_dimension(feat_list)
        if self.combination_type == "simple":
            weight_list = F.softmax(torch.sigmoid(
                self.learnable_weight[self.start:self.end]), dim=0)
        elif self.combination_type == "simple_allow_neg":
            weight_list = self.learnable_weight[self.start:self.end]
        elif self.combination_type == "gate":
            adopted_feat_list = torch.vstack(feat_list[self.start:self.end])
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(self.end - self.start, -1).T), dim=1)
        elif self.combination_type == "ori_ref":
            reference_feat = feat_list[0].repeat(self.end - self.start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self.start:self.end])))
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
        elif self.combination_type == "jk":
            reference_feat = torch.hstack(feat_list).repeat(
                self.end - self.start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self.start:self.end])))
            weight_list = F.softmax(
                torch.sigmoid(self.learnable_weight(adopted_feat_list).view(-1, self.end - self.start)), dim=1)
        else:
            raise NotImplementedError
        weighted_feat = None
        if self.combination_type == "simple" or self.combination_type == "simple_allow_neg":
            weighted_feat = one_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        elif self.combination_type in ["gate", "ori_ref", "jk"]:
            weighted_feat = two_dim_weighted_add(
                feat_list[self.start:self.end], weight_list=weight_list)
        else:
            raise NotImplementedError
        return weighted_feat
