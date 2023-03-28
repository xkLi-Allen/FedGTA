import copy
import torch

from federated.base import Server
from collections import OrderedDict


class FedGTAServer(Server):
    def __init__(self, dataset, args, device):
        super(FedGTAServer, self).__init__(dataset, args, device)

    def init_model(self):
        from gnn_model.scalable.sim_gamlp import SimGAMLP
        self.model = SimGAMLP(prop_steps=3, r=0.5, feat_dim=self.input_dim, output_dim=self.output_dim,
                                hidden_dim=self.hidden_dim, num_layers=3, dropout=self.gnn_dropout)
        self.round_id = 0
        self.lp_moment_v = None
        self.lp_moment_v = None

    def server_execute(self, server_input):
        local_models = server_input["local_models"]
        self.lp_moment_v = server_input["lp_moment_v"]
        agg_w = server_input["agg_w"]
        num_samples = len(local_models)
        server_output = [{"aggregated_model": None} for i in range(num_samples)]
        agg_client_list = {}
        for i in range(num_samples):
            agg_client_list[i] = []
            sim = torch.tensor([torch.cosine_similarity(self.lp_moment_v[i], self.lp_moment_v[j], dim=0) for j in range(num_samples)]).to(self.device)
            accept_idx = torch.where(sim > self.args.gm_alpha)
            agg_client_list[i] = accept_idx[0].tolist()
        for src, clients_list in agg_client_list.items():
            tot_w = [agg_w[i] for i in clients_list]
            aggregated_model = OrderedDict()
            cur_local_models = [local_models[client_i] for client_i in clients_list]
            cur_local_models = [a.state_dict() for a in cur_local_models]
            for it, state_dict in enumerate(cur_local_models):
                for key in state_dict.keys():
                    w_factor = tot_w[it] / sum(tot_w)
                    if it == 0:
                        aggregated_model[key] = w_factor * state_dict[key]
                    else:
                        aggregated_model[key] += w_factor * state_dict[key]
            server_output[src]["aggregated_model"] = copy.deepcopy(aggregated_model)
        self.round_id += 1
        return server_output
