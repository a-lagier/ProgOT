import torch

from progot.loader.cfg_loader import load_parser
from progot.solvers import get_solver
from progot.costs import get_cost


n = 6
a = torch.ones(n) / n
X = torch.randn((n,2))
m = 10
b = torch.ones(m) / m
Y = torch.randn((m,2))

solv = get_solver('log-sinkhorn')
h = get_cost('quadratic')

solv(a, X, b, Y, h)