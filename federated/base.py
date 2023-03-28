import copy


class Role:
    def __init__(self, dataset, args, device):
        self.args = args
        self.model_name = args.gnn_model
        self.input_dim = dataset.input_dim
        self.output_dim = dataset.output_dim
        self.device = device
        self.model = None
        self.gnn_lr = args.gnn_lr
        self.gnn_weight_decay = args.gnn_weight_decay
        self.gnn_num_epochs = args.gnn_num_epochs
        self.gnn_dropout = args.gnn_dropout
        self.hidden_dim = 64

    def init_model(self):
        raise NotImplementedError

    def set_state_dict_deep_copy(self, state_dict):
        self.model.load_state_dict(state_dict=copy.deepcopy(state_dict))


class Server(Role):
    def __init__(self, dataset, args, device):
        super(Server, self).__init__(dataset, args, device)
        self.dataset = dataset
        self.global_data = dataset.global_data
        self.num_nodes = self.global_data.num_nodes
        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def server_execute(self, server_input):
        raise NotImplementedError


class Client(Role):
    def __init__(self, client_id, dataset, args, device):
        super(Client, self).__init__(dataset, args, device)
        self.client_id = client_id
        self.local_subgraph = dataset.subgraphs[client_id]
        self.num_nodes = self.local_subgraph.num_nodes

    def init_model(self, param):
        raise NotImplementedError

    def client_execute(self, round_id):
        raise NotImplementedError

    def update_local(self, client_input):
        raise NotImplementedError

