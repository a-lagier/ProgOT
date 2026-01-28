import torch

def tensoradd(f, g):
    return f.view(-1, 1) + g.view(1, -1)

def tensormult(f, g):
    return f.view(-1, 1) * g.view(1, -1)

def softmin(z, eps=1e-1, dim=1):
    return - eps * torch.logsumexp(-z/eps, dim=dim)

def filterinf(z):
    return torch.where(z.isinf(), 0.0, z)

def rescaling(P):
    m = P.max()
    return P/m, m

def mse(y, y_pred):
    y = y.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    def filternan(z):
        import numpy as np
        return z[~np.isnan(z)]

    # return filternan(((y - y_pred) ** 2).sum(-1)).mean().item()
    return (((y - y_pred) ** 2).sum(-1)).mean().item()

def move(device, *args):
    for t in args:
        yield t.to(device)

def no_grad(*args):
    for t in args:
        yield t.detach()
    
def select_top_one(df, c):
    top_one_value = df[c].value_counts().index[0]
    return df[df[c] == top_one_value], top_one_value

def perform_pca(z, d):
    _, _, V = torch.pca_lowrank(z, q=d)
    return z @ V[:, :d]

def random_split(z, p):
    idx = torch.rand(z.shape[0]) < p
    return z[idx], z[idx.logical_not()]

def filternan(z, dim=1):
    return z[~z.isnan().any(dim)]

def get_uniform_distrib(z):
    n = z.shape[0]
    a = torch.ones(n) / n
    return a.to(z.device)