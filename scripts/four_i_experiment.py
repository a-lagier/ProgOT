import torch
import matplotlib.pyplot as plt

from progot.loader.cfg_loader import load_parser
from progot.solvers import get_solver
from progot.costs import get_cost
from progot.loader.data_loader import get_dataset
from progot.transport_render import render_coupling, render_map
from progot.utils import mse, move, no_grad

parser = load_parser()

out_dir = parser["out_dir"]
device_str = parser["device"]
device = torch.device(device_str)
seed = parser["seed"]
cost_cfg = parser["cost"]
solver_cfg = parser["solver"]
dataset_cfg = parser["dataset"]

cost_fn = get_cost(**cost_cfg)
main_solv = get_solver(**solver_cfg)
dataset = get_dataset(**dataset_cfg)

torch.manual_seed(seed)

print(dataset.get_data())