import time

from torch.optim import Adam

from tasks.base_task import BaseTask
from tasks.utils import train, evaluate, accuracy
from collections import OrderedDict
import copy


class SGLNodeClassification(BaseTask):
    def __init__(self, args, dataset, model, device):
        super(SGLNodeClassification, self).__init__(args, dataset, model, device)
        self.optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.preprocess(self.dataset.adj, self.dataset.x)
        self.model = self.model.to(self.device)
        self.labels = self.labels.to(self.device)


    def execute(self, round_id, client_id):
        best_val = 0.
        best_test = 0.
        for epoch in range(self.epochs):
            t = time.time()
            loss_train, acc_train, train_output = train(self.model, self.dataset.train_idx, self.labels, self.device,
                                                        self.optimizer, self.loss_fn)
            acc_val, acc_test, eval_output = evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                            self.labels, self.device)
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
        print('Round:{}, Client: {}'.format(round_id, client_id),
                ',Epoch: {:03d}'.format(epoch + 1),
                'loss_train: {:.4f}'.format(loss_train),
                'acc_train: {:.4f}'.format(acc_train),
                'acc_val: {:.4f}'.format(acc_val),
                'acc_test: {:.4f}'.format(acc_test),
                'time: {:.4f}s'.format(time.time() - t))
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
        return best_val, best_test, self.model, train_output, eval_output, loss_train

    def eval(self):
        acc_val, acc_test, eval_output = evaluate(self.model, self.dataset.val_idx, self.dataset.test_idx,
                                        self.labels, self.device)
        return acc_val, acc_test, eval_output

