import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='seed everything', type=int, default=2023)
parser.add_argument('--root', help='data root', type=str, default="./data")
parser.add_argument('--name', help='data name', type=str, default="PubMed")                     # Cora, CiteSeer, PubMed
parser.add_argument('--use_cuda', help='use gpu', type=bool, default=True)
parser.add_argument('--gpu_id', help='gpu id', type=int, default=0)
parser.add_argument('--num_normalize_train', help='number of normalized train times', type=int, default=1)
parser.add_argument('--eval_lr', help='learning rate: eval global or eval single client', type=float, default=1e-2)
parser.add_argument('--fl_method', help='federated learning method', type=str, default='fedgta')
parser.add_argument('--partition', help='graph partition method', type=str, default='Louvain')
parser.add_argument('--num_clients', help='number of clients', type=int, default=10)
parser.add_argument('--num_rounds', help='number of fl rounds', type=int, default=100)
parser.add_argument('--client_sample_ratio', help='client sample ratio', type=float, default=1.0)
parser.add_argument('--eval', help='eval federated train', action='store_true', default=False)
parser.add_argument('--gnn_model', help='gnn model', type=str, default="gamlp")
parser.add_argument('--gnn_num_epochs', help='number of epochs', type=int, default=3)
parser.add_argument('--gnn_lr', help='learning rate of gnn model', type=float, default=1e-2)
parser.add_argument('--gnn_weight_decay', help='weight decay of gnn model', type=float, default=5e-4)
parser.add_argument('--gnn_num_layers', help='number of gnn layers', type=int, default=3)
parser.add_argument('--gnn_dropout', help='drop out of gnn model', type=float, default=0.5)
parser.add_argument('--lp_prop', help='prop steps of label propagation', type=int, default=5)
parser.add_argument('--temperature', help='temperature of label propagation', type=int, default=20)
parser.add_argument('--moment_type', help='type of moments', type=str, default="origin")        # cora: hybrid  citeseer: hybrid  pubmed: origin
parser.add_argument('--num_moments', help='order of moments', type=int, default=13)             # cora: 6       citeseer: 20      pubmed: 13
parser.add_argument('--gm_alpha', help='gm alpha', type=float, default=0.65)                    # cora: 0.5     citeseer: 0.60    pubmed: 0.65

args = parser.parse_args()


