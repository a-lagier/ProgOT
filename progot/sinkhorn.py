import numpy as np
import matplotlib.pyplot as plt

import jax

import ott

print("ok")

def h(x):
    return np.linalg.norm(x, ord=2, axis=2)

def softmin(z, eps, axis=1):
    return -eps * np.log(np.exp(-z / eps).sum(axis))

def tensoradd(f, g):
    return f.reshape(-1, 1) + g.reshape(1, -1)


def sinkhorn(a, X, b, Y, eps, tau, f, g, h):
    C = h(X - Y)

    itr = 0
    while np.linalg.norm(np.exp((C - tensoradd(f, g)) / eps).sum(1) - a, ord=1) > tau:
        #print(itr, np.linalg.norm(np.exp((C - tensoradd(f, g)) / eps).sum(1) - a, ord=1))
        f = eps * np.log(a) - softmin(C - tensoradd(f, g), eps) + f
        g = eps * np.log(b) - softmin(C.T - tensoradd(g, f), eps) + g
        itr += 1

    return f, g, np.exp((C - (f+g)) / eps)

def plot_plan(X, Y, P, ax, thresh=1e-6, scale=1.0):
    # P is the OT plan
    n, m = P.shape[0], P.shape[1]
    Pmax = P.max()
    for i in range(n):
        for j in range(m):
            if P[i, j] / Pmax > thresh:
                ax.plot([X[i, 0], Y[j, 0]],
                [X[i, 1], Y[j, 1]],
                color='black',
                alpha=(P[i, j] / Pmax)*scale)


n = 13
m = 20
a = np.ones(n) / n
X = np.random.randn(n, 1, 2)
b = np.ones(m) / m
Y = np.random.randn(1, m, 2)
eps = 1e1
tau = 1

_, _, P = sinkhorn(a, X, b, Y, eps, tau, a*0, b*0, h)

ax = plt.gca()

plot_plan(X, Y, P, ax)
plt.show()