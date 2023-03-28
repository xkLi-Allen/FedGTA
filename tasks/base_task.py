import torch.nn as nn

class BaseTask:
    def __init__(self, args, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.labels = self.dataset.y
        self.lr = args.gnn_lr
        self.weight_decay = args.gnn_weight_decay
        self.epochs = args.gnn_num_epochs
        self.show_epoch_info = 20
        self.loss_fn = nn.CrossEntropyLoss()

    def _execute(self):
        return NotImplementedError

    def _evaluate(self):
        return NotImplementedError

    def _train(self):
        return NotImplementedError