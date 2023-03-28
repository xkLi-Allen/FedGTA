import copy
import time
import random
import numpy as np

from tasks.node_cls import SGLNodeClassification


class NormalizeTrainer:
    def __init__(self, server, clients, args, device):
        self.args = args
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))
        self.server = server
        self.num_clients = len(clients)
        self.clients = clients
        self.device = device

    def normalize_train(self):
        server = self.server
        clients = self.clients
        print("\nStart Training Federated GNN Model...")
        tm_start = time.time()
        normalize_record = {"val_acc": [], "test_acc": []}
        for _ in range(self.num_normalize_train):
            server.init_model()
            for client_id in range(self.num_clients):
                clients[client_id].init_model(server.model.state_dict())
            clients_test_acc = [0] * self.num_clients
            clients_val_acc = [0] * self.num_clients
            round_global_record = {"avg_val_acc": 0, "avg_test_acc": 0}
            for round_id in range(self.num_rounds):
                all_client_idx = list(range(self.num_clients))
                random.shuffle(all_client_idx)
                sample_num = int(len(all_client_idx) * self.client_sample_ratio)
                sample_idx = sorted(all_client_idx[:sample_num])
                num_local_nodes = [clients[idx].num_nodes for idx in sample_idx]
                server_input = {"num_local_nodes": num_local_nodes}
                client_outputs = []
                for client_id in sample_idx:
                    client_output = clients[client_id].client_execute(round_id=round_id)
                    client_outputs.append(client_output)
                for client_output in client_outputs:
                    for k, v in client_output.items():
                        if k not in server_input:
                            server_input[k] = [v]
                        else:
                            server_input[k].append(v)
                server_output = server.server_execute(server_input=server_input)
                for client_id in sample_idx:
                    clients[client_id].update_local(client_input=server_output[client_id])
                avg_val_acc = 0
                avg_test_acc = 0
                for client_id in range(self.num_clients):
                    val_acc, test_acc, _ = clients[client_id].task.eval()
                    if val_acc > clients_val_acc[client_id]:
                        clients_val_acc[client_id] = val_acc
                        clients_test_acc[client_id] = test_acc
                    avg_val_acc += (val_acc * clients[client_id].num_nodes / server.num_nodes)
                    avg_test_acc += (test_acc * clients[client_id].num_nodes / server.num_nodes)
                if avg_val_acc > round_global_record["avg_val_acc"]:
                    round_global_record["avg_val_acc"] = avg_val_acc
                    round_global_record["avg_test_acc"] = avg_test_acc
            if self.num_normalize_train == 1:
                for client_id in range(self.num_clients):
                    print("Client id: {}, Val Acc: {}, Test Acc: {}".format(client_id, round(clients_val_acc[client_id], 4), round(clients_test_acc[client_id], 4)))
            normalize_record["val_acc"].append(round_global_record["avg_val_acc"])
            normalize_record["test_acc"].append(round_global_record["avg_test_acc"])
        tm_end = time.time()
        print("Normalize Train Completed")
        print("Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(self.num_normalize_train, tm_end - tm_start))
        print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(normalize_record["val_acc"]), 4), round(np.std(normalize_record["val_acc"], ddof=1), 4), round(np.mean(normalize_record["test_acc"]), 4), round(np.std(normalize_record["test_acc"], ddof=1), 4)))
        return np.mean(normalize_record["test_acc"])

    def evaluate_data_isolate(self):
        print("\nStart Training Isolated GNN Model...")
        global_acc_test_mean = 0
        global_acc_test_std = 0
        global_acc_val_mean = 0
        global_acc_val_std = 0
        clients = copy.deepcopy(self.clients)
        server = copy.deepcopy(self.server)
        for i in range(len(clients)):
            test_acc_list = []
            val_acc_list = []
            t_total = time.time()
            for _ in range(self.num_normalize_train):
                clients[i].init_model()
                clients[i].task.epoch = self.num_rounds * self.gnn_num_epochs
                clients[i].client_execute("-")
                val_acc, test_acc, _ = clients[i].task.eval()
                test_acc_list.append(test_acc)
                val_acc_list.append(val_acc)
            print("Client ID: {}".format(i))
            print("Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(self.num_normalize_train, time.time() - t_total))
            print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(val_acc_list), 4), round(np.std(val_acc_list, ddof=1), 4), round(np.mean(test_acc_list), 4), round(np.std(test_acc_list, ddof=1), 4)))
            global_acc_test_mean += np.mean(test_acc_list) * clients[i].num_nodes / server.num_nodes
            global_acc_test_std += np.std(test_acc_list, ddof=1) * clients[i].num_nodes / server.num_nodes
            global_acc_val_mean += np.mean(val_acc_list) * clients[i].num_nodes / server.num_nodes
            global_acc_val_std += np.std(val_acc_list, ddof=1) * clients[i].num_nodes / server.num_nodes
        print("All Clients Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(global_acc_val_mean, 4), round(global_acc_val_std, 4), round(global_acc_test_mean, 4), round(global_acc_test_std, 4)))

    def evaluate_global(self):
        print("\nStart Training Global GNN Model...")
        tm_start = time.time()
        server = copy.deepcopy(self.server)
        test_acc_list = []
        val_acc_list = []
        for _ in range(self.num_normalize_train):
            server.init_model()
            val_acc, test_acc, _ = SGLNodeClassification(args=self.args, dataset=server.global_data, model=server.model, device=self.device).execute(round_id="-", client_id="-")
            test_acc_list.append(test_acc)
            val_acc_list.append(val_acc)
        tm_end = time.time()
        print("Evaluate Global Data Completed")
        print("Normalize Train: {}, Total Time Elapsed: {:.4f}s".format(self.num_normalize_train, tm_end-tm_start))
        print("Mean Val ± Std Val: {}±{}, Mean Test ± Std Test: {}±{}".format(round(np.mean(val_acc_list), 4), round(np.std(val_acc_list, ddof=1), 4), round(np.mean(test_acc_list), 4), round(np.std(test_acc_list, ddof=1), 4)))


