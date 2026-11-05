"""Original (classical) Ant Colony Optimization baseline for the TSP.

One independent run per execution. Usage (from the repository root):

    python src/aco.py --instance oliver30

Instances, colony size (m) and iteration limits are defined in src/config.py
(Table II of the paper). A log workbook (sheets "Summary" and "Iteration") is
written to results/<dataset>-<YYYYMMDD>/experiment-<N>.xlsx.
"""
import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config
from read_instance import read_coordinates

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Original ACO baseline for the TSP (one run).')
parser.add_argument('--instance', required=True,
                    choices=sorted(config.INSTANCES.keys()),
                    help='instance name defined in src/config.py')
args = parser.parse_args()

instance_cfg = config.INSTANCES[args.instance]

# Scalar parameters
tau0 = config.TAU0
alpha = config.ALPHA
beta = config.BETA
rho = config.RHO
Q = config.Q

# Colony size and iteration limit
m = instance_cfg['aco']['m']
max_iter = instance_cfg['aco']['iterations']

L = np.zeros((max_iter + 1, m + 1), dtype=float)
best_ant_of_iter = np.zeros(max_iter + 1, dtype=int)

# Real
Lmax = 0.0
iter_best_len = 0.0
best_len = 0.0

data_path = os.path.join(REPO_ROOT, 'data', instance_cfg['data_file'])


# Output folder: results/<dataset>-<date>
def get_file_stem(full_path):
    file_name_with_ext = os.path.basename(full_path)  # e.g. "Oliver30.txt"
    file_stem = os.path.splitext(file_name_with_ext)[0]  # e.g. "Oliver30"
    return file_stem


folder_name = get_file_stem(data_path) + "-" + datetime.today().strftime('%Y%m%d')
out_dir = os.path.join(REPO_ROOT, 'results', folder_name)

# Create the folder if it does not exist yet
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    run_no = 1
else:
    # Find files named "experiment-<N>.xlsx" and number this run after them
    file_list = os.listdir(out_dir)
    pattern = re.compile(r'^experiment-(\d+)\.xlsx$')
    number_list = []

    for file in file_list:
        match = pattern.match(file)
        if match:
            number_list.append(int(match.group(1)))

    if number_list:
        run_no = max(number_list) + 1
    else:
        run_no = 1

print("Execution ", out_dir, " no.", run_no)

data = read_coordinates(data_path)
n = int(data[0][0])

start_time = datetime.now()

tabu = np.zeros((m + 1, n + 1), dtype=int)
tau = np.zeros((n + 1, n + 1), dtype=float)
delta_tau = np.zeros((n + 1, n + 1), dtype=float)
tour = np.zeros(n + 1, dtype=int)
best_tour = np.zeros(n + 1, dtype=int)
tabu_init = np.zeros((m + 1, n + 1), dtype=int)


# Index conventions:
# cyclic: d[r[:-1], r[1:]]
# partial open tour: d[r[1:i], r[2:i+1]].sum()
# partial closed tour: open length + d[r[i], r[1]]

def tour_length(dt, n_tour, tour_arr):
    return dt[tour_arr[1:n_tour], tour_arr[2:n_tour + 1]].sum() + dt[tour_arr[n_tour], tour_arr[1]]


def construct_tour(dt, tour_arr):
    P = np.zeros(n + 1, dtype=float)
    PP = np.zeros(n + 1, dtype=float)
    partial_weight = np.zeros(n + 1, dtype=float)

    for n_tour in range(2, n):
        ii = tour_arr[n_tour - 1]
        Total = 0
        for jj in range(n_tour, n + 1):
            partial_weight[jj - n_tour + 1] = (tau[ii, tour_arr[jj]]**alpha) * ((1 / dt[ii, tour_arr[jj]])**beta)
            Total += partial_weight[jj - n_tour + 1]

        for jj in range(1, n - n_tour + 2):
            PP[jj] = partial_weight[jj] / Total

        # Normalizing total used for selection
        TotalM = 0
        for jj in range(1, n - n_tour + 2):
            TotalM += PP[jj]

        for jj in range(1, n - n_tour + 2):
            P[jj] = PP[jj] / TotalM

        Prand = 1 - np.random.rand()

        # Roulette selection: pick the first position whose cumulative
        # probability is >= Prand. jj is the offset (within the unvisited
        # part of the tour array) of the city to be moved to position n_tour.
        again = True
        jj = 1
        Total = 0
        while again and (jj < n - n_tour + 2):
            Total += P[jj]
            if Prand <= Total:
                again = False
            else:
                jj += 1

        swap_tmp = tour_arr[n_tour]
        tour_arr[n_tour] = tour_arr[n_tour - 1 + jj]
        tour_arr[n_tour - 1 + jj] = swap_tmp


# 1. Initialize delta_tau (rows/columns 1..n) to 0
delta_tau[1:n + 1, 1:n + 1] = 0
# 2. Initialize tau: every entry of the block is tau0
tau[1:n + 1, 1:n + 1] = tau0
# Set the main diagonal (i == j) to 0
diag_indices = np.arange(1, n + 1)
tau[diag_indices, diag_indices] = 0

# Specific to the original ACO: starting cities.
# If m == n, ant k starts from city k; otherwise each ant starts from a
# random city. City 1 is swapped into the start city's position.
tabu_init[1:m + 1, 2:n + 1] = np.arange(2, n + 1)

for k in range(1, m + 1):
    if (m == n):
        tabu_init[k, 1] = k
    else:
        tabu_init[k, 1] = n - np.random.randint(n)

    if tabu_init[k, 1] > 1:
        tabu_init[k, tabu_init[k, 1]] = 1

# Log lists
log_iter = []
log_iter_end_time = []
log_best_len = []
log_best_ant = []
log_best_tour = []

for iteration in range(1, max_iter + 1):

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            tabu[i, j] = tabu_init[i, j]

        tour[1:n + 1] = tabu[i, 1:n + 1]

        construct_tour(data, tour)
        tabu[i, 1:n + 1] = tour[1:n + 1]

    # Iteration timestamp (taken after construction, before the pheromone update)
    iter_end_time = datetime.now()

    # Compute tour lengths and keep the shortest tour
    for k in range(1, m + 1):
        L[iteration, k] = 0
        for i in range(1, n + 1):
            if i == n:
                L[iteration, k] += data[tabu[k, i], tabu[k, 1]]
            else:
                L[iteration, k] += data[tabu[k, i], tabu[k, i + 1]]
        if L[iteration, k] > Lmax:
            Lmax = L[iteration, k]

        if k == 1:
            iter_best_len = L[iteration, 1]
            best_ant_of_iter[iteration] = 1
        else:

            if L[iteration, k] < iter_best_len:
                iter_best_len = L[iteration, k]
                best_ant_of_iter[iteration] = k

        if iteration == 1 or iter_best_len < best_len:
            best_iter = iteration
            best_len = iter_best_len
            best_tour[1:n + 1] = tabu[best_ant_of_iter[iteration], 1:n + 1]

    # Record the best-so-far tour for the log
    log_iter.append(iteration)
    log_iter_end_time.append(iter_end_time)
    log_best_len.append(best_len)
    log_best_ant.append(best_ant_of_iter[best_iter])
    best_tour_joined = ', '.join(best_tour[1:n + 1].astype(str))
    log_best_tour.append(best_tour_joined)

    # Compute delta_tau for every arc and update pheromone
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            delta_tau[i, j] = 0
            for k in range(1, m + 1):
                present = False
                for pos in range(1, n + 1):
                    if pos < n:
                        if tabu[k, pos] == i and tabu[k, pos + 1] == j:
                            present = True
                    else:
                        if tabu[k, pos] == i and tabu[k, 1] == j:
                            present = True
                if present:
                    delta_tau[i, j] += Q / L[iteration, k]

            # Evaporation and deposit
            tau[i, j] = rho * tau[i, j] + delta_tau[i, j]

    # Reset delta_tau for every arc
    delta_tau[1:n + 1, 1:n + 1] = 0

    print(f"Iteration {iteration}/{max_iter}     best length so far= {best_len}")

print(f"Best length= {best_len}")

temp = '; '.join(str(x) for x in best_tour[1:n + 1])
print(f"{temp} length= {tour_length(data, n, best_tour)} found by ant [{best_ant_of_iter[best_iter]}] in iteration {best_iter}")

# Save the log workbook

best_tour_joined = ', '.join(best_tour[1:n + 1].astype(str))
# Data for the first sheet
data_sheet1 = {
    'Algorithm': 'ACO',
    'Dataset': get_file_stem(data_path),
    'No': run_no,
    'τ₀': tau0,
    'α': alpha,
    'β': beta,
    'ρ': rho,
    'Q': Q,
    'm': m,
    'maxIter': max_iter,
    'Start Time': start_time,
    'End Time': iter_end_time,
    'Best Tour Length': best_len,
    'Iteration of Best Solution': best_iter,
    'Ant of Best Solution': best_ant_of_iter[best_iter],
    'Best Tour': [best_tour_joined]
    }
df1 = pd.DataFrame(data_sheet1)


# Data for the second sheet
data_sheet2 = {
    'Iteration': log_iter,
    'Iteration End timestamp': log_iter_end_time,
    'Best-so-far Tour Length': log_best_len,
    'Ant of Best Update': log_best_ant,
    'Best Tour': log_best_tour
               }
df2 = pd.DataFrame(data_sheet2)


# Workbook file name
xlsx_file_name = out_dir + '/' + "experiment-" + str(run_no) + ".xlsx"

# Write both sheets with ExcelWriter
with pd.ExcelWriter(xlsx_file_name) as writer:
    df1.to_excel(writer, sheet_name='Summary', index=False)
    df2.to_excel(writer, sheet_name='Iteration', index=False)
