import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score

def flat(output, target):
    ypred = output.cpu().argmax(axis=1).flatten()
    yval = target.cpu().numpy().flatten()
    return yval, ypred

def confident_only_accuracy(output, target, score):
    with torch.no_grad():
        if isinstance(output, tuple): output = output[0]
        pred = torch.argmax(output, dim=1)
        pred = pred[(score == 100)]
        target = target[(score == 100)]
        return (pred == target).float().mean()

def confident_only_cm(output, target, score):
    with torch.no_grad():
        if isinstance(output, tuple): output = output[0]
        labels = output.shape[1]
        pred = torch.argmax(output, dim=1)
        pred = pred[(score == 100)]
        target = target[(score == 100)]
        return confusion_matrix(pred.cpu(), target.cpu(), labels=range(labels))

def confident_f1_score(output, target, score):
    with torch.no_grad():
        if isinstance(output, tuple): output = output[0]
        labels = output.shape[1]
        pred = torch.argmax(output, dim=1)
        pred = pred[(score == 100)]
        target = target[(score == 100)]
        return f1_score(pred.cpu(), target.cpu(), average='weighted')

def confusion_matrix_local(x, y):
    with torch.no_grad():
        if isinstance(x, tuple): x = x[0]
        ypred, yval = flat(x, y)
        return confusion_matrix(yval, ypred, labels=range(x.shape[1]))

def accuracy(output, target):
    with torch.no_grad():
        if isinstance(output, tuple): output = output[0]
        pred = torch.argmax(output, dim=1)
        return (pred == target).float().mean()
        # mask = (pred != 0).bitwise_or(target != 0)
        # assert pred.shape[0] == len(target)
        # return (pred[mask] == target[mask]).float().mean()

def f1_score_local(output, target):
    with torch.no_grad():
        output = output.cpu().numpy().argmax(axis=1)
        target = target.cpu().numpy()
        return f1_score(output.flatten(), target.flatten(), average='weighted')

