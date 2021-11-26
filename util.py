import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def save_loss(exp_name, train_list, test_list, epoch):
    h = len(train_list)
    fig = plt.figure(figsize=(20, 20))
    loss_name = ["loss", "ave_acc", "weighted_acc"]
    for i in range(h):
        ax = fig.add_subplot(h,2, 2 * i + 1)
        ax.plot(range(epoch), train_list[i])
        ax.set_title(loss_name[i] + ' for training')
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
    for i in range(h):
        ax = fig.add_subplot(h,2, 2 * i + 2)
        ax.plot(range(epoch), test_list[i])
        ax.set_title(loss_name[i] + ' for testing')
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
    # Save the full figure...
    if os.path.exists(exp_name + '_loss.png'):
        os.remove(exp_name + '_loss.png')
    fig.savefig(exp_name + '_loss.png')
    plt.close()