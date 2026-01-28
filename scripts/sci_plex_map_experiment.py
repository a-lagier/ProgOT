import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt

from progot.loader.cfg_loader import load_parser
from progot.solvers import get_solver
from progot.costs import get_cost
from progot.logger import get_logger
from progot.loader.data_loader import get_dataset
from progot.utils import move, random_split, get_uniform_distrib, filternan

parser = load_parser()

out_dir = parser['out_dir']
cfg_file = parser['config_file']
out_file = os.path.join(out_dir, cfg_file.split('.')[0] + '.log') 
device_str = parser['device']
device = torch.device(device_str)
seed = parser['seed']
nruns = parser['runs']
cost_cfg = parser['cost']
solver_cfg = parser['solver']
dataset_cfg = parser['dataset']

solver_cfg['beta_0'] = 1.0

logger = get_logger(out_file)
cost_fn = get_cost(**cost_cfg)
main_solv = get_solver(**solver_cfg, logger=logger)
dataset = get_dataset(**dataset_cfg)

logger.log_config(parser)

drugs = dataset.retained_drugs

for drug in drugs:
    X, Y = dataset.get_data(drug)

    X, Y = move(device, X, Y)
    Y_train, Y_test = random_split(Y, 0.8)

    a_train, b_train = get_uniform_distrib(X), get_uniform_distrib(Y_train)

    if solver_cfg['name'] == 'progot':
        if solver_cfg['scheduled']:
            main_solv.get_epsilons(a_train, X, b_train, Y_train, cost_fn, Y_test)
        
    _, _, _, T_prog = main_solv.solve(a_train, X, b_train, Y_train, cost_fn, log_latent_div=False)
        
    X_pred = T_prog(X)
    X_pred = filternan(X_pred, dim=1)

    a, b = get_uniform_distrib(X_pred), get_uniform_distrib(Y)

    logger.log(f'{drug}', main_solv.divergence(a, X_pred, b, Y, cost_fn).item())

    if solver_cfg['name'] == 'progot':
        main_solv.clean_potentials()
        if not solver_cfg['scheduled']:
            main_solv.set_epsilons(None)

logger.write()