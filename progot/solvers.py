import torch

from utils import tensoradd, softmin

__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None):
            raise NameError(f"{name} already registerd")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if __SOLVER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __SOLVER__[name](**kwargs)

class Solver():

    def __init__(self):
        pass

    def __call__(self, a, X, b, Y, h):
        return 

@register_solver('sinkhorn')
class Sinkhorn(Solver):

    def __init__(self, eps, tau):
        super().__init__()

        self.eps = eps
        self.tau = tau
    
    def __call__(self, a, X, b, Y, h):
        n,d = X.shape
        m,d = Y.shape

        X = X.reshape(n, 1, d)
        Y = Y.reshape(1, m, d)

        C = torch.exp(-h(X - Y) / self.eps)

        itr = 0
        while torch.linalg.norm(torch.exp((C - tensoradd(f, g)) / self.eps).sum(1) - a, ord=1) > self.tau:
            f = self.eps * torch.log(a) - softmin(C - tensoradd(f, g), self.eps) + f
            g = self.eps * torch.log(b) - softmin(C.T - tensoradd(g, f), self.eps) + g
            itr += 1

        return f, g, torch.exp((C - tensoradd(f, g)) / self.eps)

@register_solver('log-sinkhorn')
class LogSinkhorn(Solver):

    def __init__(self, eps, tau):
        super().__init__()

        self.eps = eps
        self.tau = tau
    
    def __call__(self, a, X, b, Y, h):
        n,d = X.shape
        m,d = Y.shape

        X = X.reshape(n, 1, d)
        Y = Y.reshape(1, m, d)

        C = h(X - Y)


        # TODO : implement warm startup
        f = torch.zeros(n)
        g = torch.zeros(m)

        itr = 0
        while torch.linalg.norm(torch.exp((C - tensoradd(f, g)) / self.eps).sum(1) - a, ord=1) > self.tau:
            f = self.eps * torch.log(a) - softmin(C - tensoradd(f, g), self.eps) + f
            g = self.eps * torch.log(b) - softmin(C.T - tensoradd(g, f), self.eps) + g
            itr += 1

        return f, g, torch.exp((C - tensoradd(f, g)) / self.eps)
    

@register_solver('progot')
class ProgOT(Solver):

    def __init__(self):
        super().__init__()