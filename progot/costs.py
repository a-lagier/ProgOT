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

    def __call__(self, x):
        return x

@register_cost('quadratic')
class QuadraticCost(Cost):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.linalg.norm(x, ord=2, dim=2)
