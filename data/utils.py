import torch
import random
import numpy as np
import scipy.sparse as sp

from torch import Tensor
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from louvain.community import community_louvain
from torch_geometric.utils.convert import to_networkx


def idx_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def louvain_partition(graph, num_clients):
    num_nodes = graph.number_of_nodes()
    partition = community_louvain.best_partition(graph)
    groups = []
    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    partition_groups = {group_i: [] for group_i in groups}
    for key in partition.keys():
        partition_groups[partition[key]].append(key)
    group_len_max = num_nodes // num_clients
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))
    len_dict = {}
    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {k: v for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)}
    owner_node_ids = {owner_id: [] for owner_id in range(num_clients)}
    owner_nodes_len = num_nodes // num_clients
    owner_list = [i for i in range(num_clients)]
    owner_ind = 0
    for group_i in sort_len_dict.keys():
        while len(owner_node_ids[owner_list[owner_ind]]) > owner_nodes_len:
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        k = 0
        while len(owner_node_ids[owner_list[owner_ind]]) + len(partition_groups[group_i]) > owner_nodes_len + 1:
            k += 1
            owner_ind = (owner_ind + 1) % len(owner_list)
            if k == len(owner_list):
                owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
                break
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]
    node_dict = owner_node_ids
    return node_dict

def data_partition(G, num_clients, train, val, test, partition):
    graph_nx = to_networkx(G, to_undirected=True)
    if partition == "Louvain":
        print("Conducting louvain graph partition...")
        node_dict = louvain_partition(graph=graph_nx, num_clients=num_clients)
    elif partition == "Metis":
        import metispy as metis
        print("Conducting metis graph partition...")
        node_dict = {}
        _, membership = metis.part_graph(graph_nx, num_clients)
        for client_id in range(num_clients):
            client_indices = np.where(np.array(membership) == client_id)[0]
            client_indices = list(client_indices)
            node_dict[client_id] = client_indices
    else:
        raise ValueError(f"No such partition method: '{partition}'.")
    subgraph_list = construct_subgraph_dict_from_node_dict(
        G=G,
        num_clients=num_clients,
        node_dict=node_dict,
        graph_nx=graph_nx,
        train=train,
        val=val,
        test=test
    )
    return subgraph_list

def construct_subgraph_dict_from_node_dict(num_clients, node_dict, G, graph_nx, train, val, test):
    subgraph_list = []
    for client_id in range(num_clients):
        num_local_nodes = len(node_dict[client_id])
        local_node_idx = [idx for idx in range(num_local_nodes)]
        random.shuffle(local_node_idx)
        train_size = int(num_local_nodes * train)
        val_size = int(num_local_nodes * val)
        test_size = int(num_local_nodes * test)
        train_idx = local_node_idx[: train_size]
        val_idx = local_node_idx[train_size: train_size + val_size]
        test_idx = local_node_idx[train_size + val_size:]
        local_train_idx = idx_to_mask(train_idx, size=num_local_nodes)
        local_val_idx = idx_to_mask(val_idx, size=num_local_nodes)
        local_test_idx = idx_to_mask(test_idx, size=num_local_nodes)
        map_train_idx = []
        map_val_idx = []
        map_test_idx = []
        map_train_idx += [node_dict[client_id][idx] for idx in train_idx]
        map_val_idx += [node_dict[client_id][idx] for idx in val_idx]
        map_test_idx += [node_dict[client_id][idx] for idx in test_idx]
        global_train_idx = idx_to_mask(map_train_idx, size=G.y.size(0))
        global_val_idx = idx_to_mask(map_val_idx, size=G.y.size(0))
        global_test_idx = idx_to_mask(map_test_idx, size=G.y.size(0))
        node_idx_map = {}
        edge_idx = []
        for idx in range(num_local_nodes):
            node_idx_map[node_dict[client_id][idx]] = idx
        edge_idx += [(node_idx_map[x[0]], node_idx_map[x[1]]) for x in
                     graph_nx.subgraph(node_dict[client_id]).edges]
        edge_idx += [(node_idx_map[x[1]], node_idx_map[x[0]]) for x in
                     graph_nx.subgraph(node_dict[client_id]).edges]
        edge_idx_tensor = torch.tensor(edge_idx, dtype=torch.long).T
        subgraph = Data(x=G.x[node_dict[client_id]],
                        y=G.y[node_dict[client_id]],
                        edge_index=edge_idx_tensor)
        subgraph.adj = sp.coo_matrix((torch.ones([len(edge_idx_tensor[0])]), (edge_idx_tensor[0], edge_idx_tensor[1])),
                                     shape=(num_local_nodes, num_local_nodes))
        subgraph.row, subgraph.col, subgraph.edge_weight = subgraph.adj.row, subgraph.adj.col, subgraph.adj.data
        if isinstance(subgraph.adj.row, Tensor) or isinstance(subgraph.adj.col, Tensor):
            subgraph.adj = csr_matrix((subgraph.edge_weight.numpy(), (subgraph.row.numpy(), subgraph.col.numpy())),
                                      shape=(subgraph.num_nodes, subgraph.num_nodes))
        else:
            subgraph.adj = csr_matrix((subgraph.edge_weight, (subgraph.row, subgraph.col)),
                                      shape=(subgraph.num_nodes, subgraph.num_nodes))
        subgraph.train_idx = local_train_idx
        subgraph.val_idx = local_val_idx
        subgraph.test_idx = local_test_idx
        subgraph.global_train_idx = global_train_idx
        subgraph.global_val_idx = global_val_idx
        subgraph.global_test_idx = global_test_idx
        subgraph_list.append(subgraph)
        print("Client: {}\tTotal Nodes: {}\tTotal Edges: {}\tTrain Nodes: {}\tVal Nodes: {}\tTest Nodes\t{}".format(
            client_id + 1, subgraph.num_nodes, subgraph.num_edges, train_size, val_size, test_size))
    return subgraph_list

