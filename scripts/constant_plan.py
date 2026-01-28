import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import ot

from progot.loader.cfg_loader import load_parser
from progot.solvers import get_solver
from progot.costs import get_cost
from progot.logger import get_logger
from progot.loader.data_loader import get_dataset
from progot.utils import move, no_grad, get_uniform_distrib

parser = load_parser()

out_dir = parser["out_dir"]
cfg_file = parser['config_file']
out_file = os.path.join(out_dir, cfg_file.split('.')[0] + '.log') 
device_str = parser["device"]
device = torch.device(device_str)
seed = parser["seed"]
cost_cfg = parser["cost"]
solver_cfg = parser["solver"]
dataset_cfg = parser["dataset"]

logger = get_logger(out_file)
cost_fn = get_cost(**cost_cfg)
main_solv = get_solver(**solver_cfg)
dataset = get_dataset(**dataset_cfg)

torch.manual_seed(seed)
logger.log_config(parser)

X_train, X_test, Y_train, Y_test = dataset.get_data()
a = get_uniform_distrib(X_train)

a, X_train, X_test, Y_train, Y_test = move(device, a, X_train, X_test, Y_train, Y_test)
a, X_train, X_test, Y_train, Y_test = no_grad(a, X_train, X_test, Y_train, Y_test)

# Entropic solver
# main_solv.get_epsilons(a, X_train, a, Y_train, cost_fn, Y_test)
_, _, _, _ = main_solv.solve(a, X_train, a, Y_train, cost_fn, log_latent_div=False, log_latent_plan=True)
print("Entropic solving done!")

C = cost_fn(X_train, Y_train)

# Exact solver
res = ot.solve(C, a, a)
P_exact = res.plan
print("Exact solving done!")
print("Cost of exact plan :", (C * P_exact).sum().item())

plan_err = []
for i,latent_plan in enumerate(main_solv.plans):
    logger.log(i, "| Cost of entropic plan", (C * latent_plan).sum().item())
    logger.log(i, "| TV of entropic plan", 1/2 * (latent_plan - P_exact).abs().sum().item())
    plan_err.append(1/2 * (latent_plan - P_exact).abs().sum().item())

logger.write()