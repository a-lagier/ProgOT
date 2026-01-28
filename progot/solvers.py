import torch
import numpy as np
from functools import partial

from .utils import tensoradd, softmin, mse

def interpolate(time_steps, start=0.0, end=1.0, mode = 'linear'):
    # TODO : implement other interpolation method (cosine)
    if mode == 'linear':
        out = np.linspace(start, end, time_steps)
        return torch.from_numpy(out).float()
    elif mode == 'constant-speed':
        out = torch.from_numpy(np.arange(time_steps))
        return 1./ (time_steps - out + 2)
    elif mode == 'decreased-speed':
        return  torch.ones(time_steps) / np.e
    elif mode == 'increased-speed':
        out = torch.from_numpy(np.arange(time_steps))
        return (2 * out - 1) / ((time_steps + 1) ** 2 - (out - 1) ** 2)
    else:
        raise ValueError(f"Unknown mode for interpolation mode={mode}")

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
    
    def transport_cost(self, P, C):
        return (P * C).sum()

    def entropy(self, P):
        log_P = P.log()
        log_P[log_P.isinf()] = 0.
        return - (P * log_P).sum()

    def marginal_sat(self, a, b, P):
        sat_1 = torch.linalg.norm(P.sum(dim=1) - a, ord=1)
        sat_2 = torch.linalg.norm(P.T.sum(dim=1) - b, ord=1)
        return sat_1 + sat_2

    def divergence(self, a, X, b, Y, h):
        eps_D = 1/20 * h(Y, Y).mean()

        _, _, P_xy, _ = self.solve(a, X, b, Y, h, record_potential=False)
        ot_xy = self.get_objective_value(P_xy, h(X, Y), eps_D)

        _, _, P_xx, _ = self.solve(a, X, a, X, h, record_potential=False)
        ot_xx = self.get_objective_value(P_xx, h(X, X), eps_D)
        
        _, _, P_yy, _ = self.solve(b, Y, b, Y, h, record_potential=False)
        ot_yy = self.get_objective_value(P_yy, h(Y, Y), eps_D)

        return ot_xy - (ot_xx + ot_yy) / 2

    def get_objective_value(self, P, C, eps):
        H_P = - (P * (P + 1e-12).log()).sum()
        return (P * C).sum() + eps * H_P
    

@register_solver('log-sinkhorn')
class LogSinkhorn(Solver):

    def __init__(self, **kwargs):
        super().__init__()

        self.eps = None
        self.tau = None
    
    def solve(self, a, X, b, Y, h, eps = None, tau = None, f_init = None, g_init = None, **kwargs):
        C = h(X, Y)

        # TODO : change this
        if self.eps is None:
            self.eps = 1/20 * C.mean()
        
        if self.tau is None:
            self.tau = 1e-3
        
        if eps:
            self.eps = eps

        if tau:
            self.tau = tau

        f = torch.zeros_like(a, device=a.device) if f_init is None else f_init
        g = torch.zeros_like(b, device=b.device) if g_init is None else g_init

        itr = 0
        itr_limit = 10000
        while self.get_error(C, f, g, a, axis=1) > self.tau and itr < itr_limit:
            f = self.eps * torch.log(a) + softmin(C - g.view(1, -1), self.eps, dim=1)
            g = self.eps * torch.log(b) + softmin(C.T - f.view(1, -1), self.eps, dim=1)
            itr += 1
        if itr == itr_limit:
            print('Sinkhorn iterations limit hit !')

        P = torch.exp((tensoradd(f, g) - C) / self.eps)
        T_prog = partial(transport, b, Y, h, [g], [self.eps], [1])

        return f, g, P, T_prog
    
    def get_error(self, C, f, g, marginal, axis = 1):
        return torch.linalg.norm(torch.exp((tensoradd(f, g) - C) / self.eps).sum(axis) - marginal, ord=1)

@register_solver('progot')
class ProgOT(Solver):

    def __init__(self, alphas_cfg, taus_cfg, epsilon_scalers, beta_0, K, sink_solver_name, logger=None, scheduled=True, **kwargs):
        super().__init__()

        self.K = K
        self.alphas = interpolate(K, **alphas_cfg)
        self.taus = interpolate(K, **taus_cfg)
        self.epsilon_scalers = epsilon_scalers
        self.beta_0 = beta_0

        if not scheduled:
            self.epsilons = [None] * K
            self.unscheduled_scaling = kwargs.get('unscheduled_scaling', 1.0)

        self.potentials = []
        self.plans = []

        self.sink_solver = get_solver(sink_solver_name)

        self.logger = logger

    def solve(self, a, X, b, Y, h, record_potential=True, log_latent_div=False, log_latent_plan=False):
        X = X.clone()

        f = torch.zeros_like(a, device=a.device)
        g = torch.zeros_like(b, device=b.device)

        for k in range(self.K):
            if log_latent_div:
                self.logger.log(f'itr {k} | Sink Div {round(self.divergence(a, X, b, Y, h).item(), 4)}')

            eps, alpha, tau = self.get_parameters(k)
            f_init, g_init = (1 - alpha) * f, (1 - alpha) * g

            if eps is None:
                self.epsilons[k] = 1/20 * h(X, Y).mean()
                eps = self.unscheduled_scaling * self.epsilons[k]

            f, g, P, _ = self.sink_solver.solve(a, X, b, Y, h, f_init=f_init, g_init=g_init, eps=eps, tau=tau)
            Q = (1. / P.sum(1, keepdim=True)) * P

            grad_cost = h.grad(X, Y)
            T = (Q.unsqueeze(-1) * grad_cost).sum(1)
            Z = h.grad_dual(T)
            X -= alpha * Z

            
            if log_latent_plan:
                self.plans.append(P)

            if record_potential:
                self.potentials.append(g)

        T_prog = partial(transport, b, Y, h, self.potentials, self.epsilons, self.alphas)
        return f, g, P, T_prog

    def get_epsilons(self, a, X, b, Y, h, Y_test):
        eps_0 = 1/20 * h(X, Y).mean()
        sigma = 1/20 * h(Y, Y).mean()
        # TODO : find a way to fix the tau
        tau = 1e-4

        errors = [] 
        for s in self.epsilon_scalers:
            eps = s * sigma
            _, _, _, T = self.sink_solver.solve(b, Y, b, Y, h, eps=eps, tau=tau)
            err = mse(Y_test, T(Y_test))
            errors.append(err)

        opt_scaling = self.epsilon_scalers[np.argmin(errors)]
        eps_1 = opt_scaling * sigma

        t_k = 1. - torch.cumprod(1. - self.alphas, dim=0)
        t_k = t_k.to(eps_1.device)

        self.epsilons = (1. - t_k) * self.beta_0 * eps_0 + t_k * eps_1
    
    def divergence(self, a, X, b, Y, h):
        eps_D = 1/20 * h(Y, Y).mean()

        _, _, P_xy, _ = self.sink_solver.solve(a, X, b, Y, h, eps=eps_D, tau=1e-3, record_potential=False)
        ot_xy = self.get_objective_value(P_xy, h(X, Y), eps_D)

        _, _, P_xx, _ = self.sink_solver.solve(a, X, a, X, h, eps=eps_D, tau=1e-3, record_potential=False)
        ot_xx = self.get_objective_value(P_xx, h(X, X), eps_D)
        
        _, _, P_yy, _ = self.sink_solver.solve(b, Y, b, Y, h, eps=eps_D, tau=1e-3, record_potential=False)
        ot_yy = self.get_objective_value(P_yy, h(Y, Y), eps_D)

        return ot_xy - (ot_xx + ot_yy) / 2 + eps_D / 2 * (a.sum() - b.sum()) ** 2

    def set_epsilons(self, eps):
        self.epsilons = [eps for _ in range(self.K)]

    def clean_potentials(self):
        self.potentials = []

    def get_parameters(self, k):
        return (self.epsilons[k], self.alphas[k], self.taus[k])


def transport(b, Y, h, g_list, eps_list, alpha_list, x):
    K = len(g_list)

    y = x
    for k in range(K):
        g_k = g_list[k]
        eps_k = eps_list[k]
        alpha_k = alpha_list[k]

        p = b.view(-1, 1) * torch.exp((g_k.unsqueeze(-1) - h(Y, y)) / eps_k)
        p = p / p.sum(0, keepdim=True)

        delta = h.grad(Y, y, first=False)
        t = (p.unsqueeze(-1) * delta).sum(0)
        z = h.grad_dual(t)
        y = y - alpha_k * z
    return y