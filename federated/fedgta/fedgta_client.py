import math
import torch
import torch.nn.functional as F

from federated.base import Client
from gnn_model.gm import compute_moment
from tasks.node_cls import SGLNodeClassification
from gnn_model.scalable.sim_gamlp import SimGAMLP
from gnn_model.label_propagation import NonParaLP
from utils import sparse_mx_to_torch_sparse_tensor


def info_entropy_rev(vec, num_neig, eps=1e-8):
    return (num_neig.sum()) * vec.shape[1] * math.exp(-1) + torch.sum(torch.multiply(num_neig, torch.sum(torch.multiply(vec, torch.log(vec+eps)), dim=1)))


class FedGTAClient(Client):
    def __init__(self, client_id, dataset, args, device):
        super(FedGTAClient, self).__init__(client_id, dataset, args, device)

    def init_model(self, param=None):
        print(f"Client {self.client_id} initialize, "
              f"please wait (depending on the size of subgraph)...")
        self.model = SimGAMLP(prop_steps=3, r=0.5, feat_dim=self.input_dim, output_dim=self.output_dim,
                                hidden_dim=self.hidden_dim, num_layers=3, dropout=self.gnn_dropout)
        if param is not None:
            self.set_state_dict_deep_copy(param)
        self.task = SGLNodeClassification(args=self.args, dataset=self.local_subgraph, model=self.model,
                                         device=self.device)
        self.LP = NonParaLP(prop_steps=self.args.lp_prop, num_class=self.output_dim, subgraph=self.local_subgraph, device=self.device)
        self.num_neig = torch.sparse.sum(
            sparse_mx_to_torch_sparse_tensor(self.local_subgraph.adj), dim=1).to_dense().to(self.device)

    def client_execute(self, round_id):
        client_output = {}
        _, _, local_model, _, eval_output, loss_train = self.task.execute(round_id, self.client_id)
        client_output["local_models"] = local_model
        eval_output = F.softmax(eval_output.detach(), dim=1)
        self.LP.preprocess(soft_label=eval_output)
        lp_labels, lp_labels_dis = self.LP.propagate()
        lp_moment_v = compute_moment(x=lp_labels, num_moments=self.args.num_moments, dim="v", moment_type=self.args.moment_type)
        client_output["lp_moment_v"] = lp_moment_v.view(-1)
        client_output["agg_w"] = info_entropy_rev(lp_labels_dis, self.num_neig)
        return client_output

    def update_local(self, client_input):
        aggregated_model = client_input["aggregated_model"]
        self.set_state_dict_deep_copy(aggregated_model)
