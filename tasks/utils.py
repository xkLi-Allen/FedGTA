from torchviz import make_dot


def view_grad(input, model_list=None):
    if model_list is not None:
        model_list = [list(model.named_parameters()) for model in model_list]
        model_ = list()
        for i in model_list:
            model_ += i
        make_dot(input, dict(model_)).view()
    else:
        make_dot(input).view()

def train(model, train_idx, labels, device, optimizer, loss_fn):
    model.train()
    train_output = model.forward(train_idx, device)
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train, train_output

def evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    output = model.forward(range(len(val_idx)), device)
    acc_val = accuracy(output[val_idx], labels[val_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    return acc_val, acc_test, output

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()
