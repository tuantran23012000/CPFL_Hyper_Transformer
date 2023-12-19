import torch.nn.functional as F
from torch import nn

def nll(pred, gt, val=False, ignore_index = -100):
    if val:
        return F.nll_loss(pred, gt, size_average=False,ignore_index = ignore_index)
    else:
        return F.nll_loss(pred, gt,ignore_index = ignore_index)
def ce(pred, gt):
    return F.cross_entropy(pred, gt)
def l1_norm(pred, gt):
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1)
    return torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
def dot_product(pred, gt):
    binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1)
    return 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
def mse(pred, gt):
    return F.mse_loss(pred, gt, reduction="none").mean(0)

def get_loss(params, configs):
    if configs["name_exp"] == 'celebA':
        loss_fn = {}
        for t in params["tasks"]:
            loss_fn[t] = nll
        return loss_fn
    elif configs["name_exp"] == 'mnist':
        loss_fn = {}
        for t in params["tasks"]:
            loss_fn[t] = ce
        return loss_fn
    elif configs["name_exp"] == 'nlp':
        loss_fn = {}
        loss_fn["1"] = mse
        loss_fn["2"] = ce
        return loss_fn
    elif configs["name_exp"] == 'nuyv2':
        loss_fn = {}
        loss_fn["semantic"] = nll
        loss_fn["depth"] = l1_norm
        return loss_fn
    elif configs["name_exp"] == 'sarcos':
        loss_fn = {}
        for t in params["tasks"]:
            loss_fn[t] = mse
        return loss_fn