[x] Find datasets + implement dataset loader
[x] Implement metrics
[x] implement progot algo
[x] implement eps-scheduler
[x] implement cfg
[ ] implement loggers
[x] visualize plan/transport
[x] enable gpu usage
[x] Emmanuel Quemener
[ ] Synthetic experiment (scheduler comparison, number of iterations, impact of scheduler ??)
[ ] Sci-plex experiment (for each drug show and dimension (8, 16, 32) show divergence + basic sinkhorn results)
[ ] 4i experiment (top-6 drugs reimplement the cost/entropy experiments) (!!!!dataset unavailable!!!!)

Plan for experiments:
1. Map estimator
    1. Eps-scheduler confirmation
    2. Synthetic and Real data (ProgOT vs. Sinkhorn)
2. Plan estimators
    1. Eps depency on synthetic (maybe vs. Sinkhorn)
    2. CIFAR10 or synthetic (talk about scalability ??)
    3. Try on Sci-plex dataset to see results (entropy/cost) (fewer diversity)

Notes:
    - transport only works starightforward with the original X. For new X, use barycenters methods...
    Synthetic Data:
        - https://github.com/iamalexkorotin/Wasserstein2Benchmark/tree/main
    Real Data:
        - https://github.com/cole-trapnell-lab/sci-plex

Sci-plex:
    - file pdata : top_obligo is the drug that makes most reactions (read experience to see protocol)