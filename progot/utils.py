import torch


def tensoradd(f, g):
    return f.view(-1, 1) + g.view(1, -1)


def softmin(z, eps=1e-1, dim=1):
    return - eps * (-z/eps).exp().sum(dim).log()