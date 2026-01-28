import torch
import matplotlib.pyplot as plt


def render_map(X, Y, ax):
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    for i in range(X.shape[0]):
        ax.arrow(X[i, 0], X[i, 1],  Y[i, 0] - X[i, 0], Y[i, 1] - X[i, 1])
    ax.scatter(X[:, 0], X[:, 1], c='green', label='X_test')
    ax.scatter(Y[:, 0], Y[:, 1], c='yellow', label='Y_pred')
    return ax

def render_coupling(X, Y, P, ax, thresh=1e-6, scale=1.0):
    # datasets X, Y must contain 2D vectors
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()

    ax.grid()

    n,m = P.shape
    pmax = P.max().item()
    for i in range(min(n, 50)):
        for j in range(min(m, 50)):
            if P[i, j] / pmax > thresh:
                ax.plot([X[i, 0], Y[j, 0]],
                [X[i, 1], Y[j, 1]],
                color='black',
                alpha=(P[i, j].item() / pmax)*scale)
    
    ax.scatter(X[:, 0], X[:, 1], color='blue', label='source')
    ax.scatter(Y[:, 0], Y[:, 1], color='red', label='target')
    ax.legend()

    return ax