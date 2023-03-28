from gnn_model.scalable.base_model import SimBaseSGModel
from gnn_model.scalable.simple_models import SimMultiLayerPerceptron
from operators.graph_operator.symmetrical_simgraph_laplacian_operator import SymSimLaplacianGraphOp
from operators.message_operator.sim_message_operator.sim_learnable_weighted_messahe_op import SimLearnableWeightedMessageOp

class SimGAMLP(SimBaseSGModel):
    def __init__(self, prop_steps, r, feat_dim, output_dim, hidden_dim, num_layers, dropout):
        super(SimGAMLP, self).__init__(prop_steps, feat_dim, output_dim)
        self.pre_graph_op = SymSimLaplacianGraphOp(prop_steps, r=r)
        self.pre_msg_op = SimLearnableWeightedMessageOp(0, prop_steps + 1, "jk", prop_steps, feat_dim)
        self.base_model = SimMultiLayerPerceptron(feat_dim, hidden_dim, num_layers, output_dim, dropout)