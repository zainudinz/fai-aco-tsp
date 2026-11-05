"""Experiment configuration (Table II of the paper).

Parameters shared by both algorithms: tau0 = 0.05, alpha = 1, beta = 5,
rho = 0.99, Q = 1. Instance-specific colony size (m) and iteration limit
differ between the original ACO and FAI-ACO.
"""

TAU0 = 0.05   # initial pheromone
ALPHA = 1.0   # pheromone exponent
BETA = 5.0    # heuristic exponent
RHO = 0.99    # pheromone persistence factor (as used in the pheromone update)
Q = 1         # pheromone deposit constant

# instance -> data file and per-algorithm (m, iteration limit)
INSTANCES = {
    'oliver30': {
        'data_file': 'Oliver30.txt',
        'aco':     {'m': 30,  'iterations': 250},
        'fai_aco': {'m': 3,   'iterations': 250},
    },
    'eil51': {
        'data_file': 'eil51.txt',
        'aco':     {'m': 51,  'iterations': 5000},
        'fai_aco': {'m': 5,   'iterations': 500},
    },
    'kroA100': {
        'data_file': 'kroA100.txt',
        'aco':     {'m': 100, 'iterations': 5000},
        'fai_aco': {'m': 10,  'iterations': 500},
    },
}
