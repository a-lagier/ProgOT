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
from progot.transport_render import render_coupling, render_map
from progot.utils import mse, move, no_grad, get_uniform_distrib

parser = load_parser()

out_dir = parser['out_dir']
cfg_file = parser['config_file']
out_file = os.path.join(out_dir, cfg_file.split('.')[0] + '.log') 
device_str = parser['device']
device = torch.device(device_str)
seed = parser['seed']
cost_cfg = parser['cost']
solver_cfg = parser['solver']
dataset_cfg = parser['dataset']

logger = get_logger(out_file)
cost_fn = get_cost(**cost_cfg)
main_solv = get_solver(**solver_cfg, logger=logger)
dataset = get_dataset(**dataset_cfg)

torch.manual_seed(seed)
logger.log_config(parser)

X_train, X_test, Y_train, Y_test = dataset.get_data()
a = get_uniform_distrib(X_train)
a, X_train, X_test, Y_train, Y_test = move(device, a, X_train, X_test, Y_train, Y_test)
a, X_train, X_test, Y_train, Y_test = no_grad(a, X_train, X_test, Y_train, Y_test)

if solver_cfg['name'] == 'progot':
    if solver_cfg['scheduled']:
        main_solv.get_epsilons(a, X_train, a, Y_train, cost_fn, Y_test)

_, _, P, T_prog = main_solv.solve(a, X_train, a, Y_train, cost_fn, log_latent_div=True)

logger.log(mse(Y_test, T_prog(X_test)))

logger.write()