# ProgOT
Reimplementation of ProgOT [[Kassraie et al. 2025]](https://arxiv.org/abs/2406.05061), a new class of Entropic Optimal Transport (EOT) solvers that aims to lower the effect of the entropic regularization parameter $\varepsilon$. The repo provides a complete implementation of the method and tools to emulate the differents experiments performed in the original paper.

## Technical startup
Create a conda environment by running
```
conda create -n progot python=3.9
conda activate
pip install -r requirements.txt
```

## Supplementary Datasets / Repo
To have access to the synthetic dataset Mix3ToMix10 see https://github.com/iamalexkorotin/Wasserstein2Benchmark.git.

To have access to the single-cell dataset Sci-Plex see https://github.com/cole-trapnell-lab/sci-plex. When using this dataset, the code automatically reprocess the data from scratch to extract source/target population for OT experiment.

## Running experiments
For example, if you want to run the map Sci-Plex experiments with $d=16$ and constant speed $\alpha$-scheduling simply run
```
python scripts/sci_plex_map_experiment.py --cfg configs/ot-sci-plex-constant-16.yaml
```

### Authors
Alexandre Lagier