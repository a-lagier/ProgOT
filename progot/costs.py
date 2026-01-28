import torch

__COST__ = {}

def register_cost(name: str):
    def wrapper(cls):
        if __COST__.get(name, None):
            raise NameError(f"{name} already registerd")
        __COST__[name] = cls
        return cls
    return wrapper

def get_cost(name: str, **kwargs):
    if __COST__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __COST__[name](**kwargs)

class Cost():

    def __init__(self):
        pass

    def shape_input(self, x, y):
        return x[:, None, ...], y[None, ...]

    def __call__(self, x, y):
        return x - y
    
    def convex_dual(self, x, y):
        return x - y

@register_cost('quadratic')
class QuadraticCost(Cost):

    def __init__(self):
        super().__init__()

    def __call__(self, x, y=None):
        if y is None:
            return 1/2 * torch.linalg.norm(x, ord=2, dim=2) ** 2
        x, y = self.shape_input(x, y)
        return 1/2 * torch.linalg.norm(x - y, ord=2, dim=2) ** 2
    
    def grad(self, x, y=None, first=True):
        if y is None:
            return self.grad1(x, y)
        x, y = self.shape_input(x, y)
        if first:
            return self.grad1(x, y)
        return self.grad2(x, y)

    def grad1(self, x, y=None):
        if y is None:
            return x
        return (x - y)

    def grad2(self, x, y=None):
        if y is None:
            return x
        return (y - x)

    def convex_dual(self, x, y=None):
        return self(x, y)

    def grad_dual(self, x, y=None):
        return self.grad(x, y)