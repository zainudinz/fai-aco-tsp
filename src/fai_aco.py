"""Flying-Arc Insertion Ant Colony Optimization (FAI-ACO) for the TSP.

One independent run per execution. Usage (from the repository root):

    python src/fai_aco.py --instance oliver30

Instances, colony size (m) and iteration limits are defined in src/config.py
(Table II of the paper). A log workbook (sheets "Summary" and "Iteration") is
written to results/<dataset>-<YYYYMMDD>/experiment-<N>.xlsx.
"""
import argparse
import os
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config
from read_instance import read_coordinates

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='FAI-ACO for the TSP (one run).')
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
m = instance_cfg['fai_aco']['m']
max_iter = instance_cfg['fai_aco']['iterations']

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


# Index conventions:
# cyclic: d[r[:-1], r[1:]]
# partial open tour: d[r[1:i], r[2:i+1]].sum()
# partial closed tour: open length + d[r[i], r[1]]

def tour_length(dt, n_tour, tour_arr):
    return dt[tour_arr[1:n_tour], tour_arr[2:n_tour + 1]].sum() + dt[tour_arr[n_tour], tour_arr[1]]


def construct_tour(dt, tour_arr, first):
    remaining = np.zeros(n + 1, dtype=int)
    cand_tour = np.zeros(n + 1, dtype=int)
    rotated = np.zeros(n + 1, dtype=int)
    best_cand_tour = np.zeros(n + 1, dtype=int)
    P = np.zeros(n + 1, dtype=float)
    PP = np.zeros(n + 1, dtype=float)

    n_remaining = n
    n_tour = 1

    def rotate_to_start(target_tour, new_start_city):
        nonlocal n_tour
        """
        Rotate the partial tour (of length n_tour) in place so that
        new_start_city becomes the first element (index 1).
        """
        # Step 1: isolate the active part of the tour
        active = target_tour[1: n_tour + 1].copy()

        # Step 2: find the position of the new start city
        # (assumes new_start_city is always in the active part)
        position = np.where(active == new_start_city)[0][0]

        # Step 3: rotate the active part
        shift = -position
        active_rotated = np.roll(active, shift)

        # Steps 4 and 5: update target_tour in place
        target_tour[1: n_tour + 1] = active_rotated
        target_tour[0] = target_tour[n_tour]

    def shift_down(x, start):
        nonlocal n_remaining

        # 1. Shift the elements left (vectorized).
        x[start:n_remaining] = x[start + 1:n_remaining + 1]

        # 2. Set the now-unused last element to 0.
        x[n_remaining] = 0

        # 3. Reduce the effective range.
        n_remaining -= 1

    def shift_up(x, start):
        nonlocal n_tour
        n_tour += 1
        if start < n_tour:
            # Shift the block [start .. n_tour-1] to [start+1 .. n_tour]
            x[start + 1: n_tour + 1] = x[start: n_tour]

    remaining[1:n + 1] = np.arange(1, n + 1)
    remaining[n + 1:] = 0

    tour_arr[1] = first
    shift_down(remaining, tour_arr[1])

    # random
    pos_min = n_remaining - np.random.randint(0, n_remaining)

    n_tour = 2
    tour_arr[2] = remaining[pos_min]
    shift_down(remaining, pos_min)
    pos_min = n_remaining - np.random.randint(0, n_remaining)

    n_tour = 3
    tour_arr[3] = remaining[pos_min]
    shift_down(remaining, pos_min)
    tour_arr[4:] = 0

    for i in range(4, n + 1):
        # Weight of every arc of the closed partial tour
        Total = (tau[tour_arr[n_tour], tour_arr[1]]**alpha) * ((1 / dt[tour_arr[n_tour], tour_arr[1]])**beta)
        for jj in range(1, n_tour):
            Total += (tau[tour_arr[jj], tour_arr[jj + 1]]**alpha) * ((1 / dt[tour_arr[jj], tour_arr[jj + 1]])**beta)

        # Complementary probability of each arc
        for jj in range(1, n_tour):
            PP[jj] = 1 - ((tau[tour_arr[jj], tour_arr[jj + 1]]**alpha) * ((1 / dt[tour_arr[jj], tour_arr[jj + 1]])**beta)) / Total
        PP[n_tour] = 1 - ((tau[tour_arr[n_tour], tour_arr[1]]**alpha) * ((1 / dt[tour_arr[n_tour], tour_arr[1]])**beta)) / Total

        # Normalizing total used for selection
        TotalM = 0
        for jj in range(1, n_tour + 1):
            TotalM += PP[jj]

        for jj in range(1, n_tour + 1):
            P[jj] = PP[jj] / TotalM

        Prand = 1 - np.random.rand()

        # Roulette selection: pick the first position whose cumulative
        # probability is >= Prand. release_pos is the position in the partial
        # tour of the arc to be released (opened).
        again = True
        release_pos = 1
        Total = 0
        while again and (release_pos < n_tour):
            Total += P[release_pos]
            if Prand <= Total:
                again = False
            else:
                release_pos += 1

        # pos_min is the position in `remaining` of the city to be inserted
        # (endpoint-sum insertion cost)
        pos_min = 1
        if release_pos < n_tour:
            ins_cost = dt[tour_arr[release_pos], remaining[1]] + dt[tour_arr[release_pos + 1], remaining[1]] - dt[tour_arr[release_pos], tour_arr[release_pos + 1]]
            for jj in range(2, n_remaining + 1):
                if ins_cost > dt[tour_arr[release_pos], remaining[jj]] + dt[tour_arr[release_pos + 1], remaining[jj]] - dt[tour_arr[release_pos], tour_arr[release_pos + 1]]:
                    ins_cost = dt[tour_arr[release_pos], remaining[jj]] + dt[tour_arr[release_pos + 1], remaining[jj]] - dt[tour_arr[release_pos], tour_arr[release_pos + 1]]
                    pos_min = jj
        else:
            ins_cost = dt[tour_arr[release_pos], remaining[1]] + dt[tour_arr[1], remaining[1]] - dt[tour_arr[release_pos], tour_arr[1]]
            for jj in range(2, n_remaining + 1):
                if ins_cost > dt[tour_arr[release_pos], remaining[jj]] + dt[tour_arr[1], remaining[jj]] - dt[tour_arr[release_pos], tour_arr[1]]:
                    ins_cost = dt[tour_arr[release_pos], remaining[jj]] + dt[tour_arr[1], remaining[jj]] - dt[tour_arr[release_pos], tour_arr[1]]
                    pos_min = jj

        shift_up(tour_arr, release_pos + 1)
        tour_arr[release_pos + 1] = remaining[pos_min]
        tour_arr[0] = tour_arr[n_tour]
        shift_down(remaining, pos_min)
        inserted_city = tour_arr[release_pos + 1]
        rotate_to_start(tour_arr, inserted_city)

        # Reordering-based local search
        best_cand_len = tour_length(dt, n_tour, tour_arr)

        # copy tour_arr -> best_cand_tour
        best_cand_tour[:n_tour + 1] = tour_arr[:n_tour + 1]

        # June 21, 2025: cyclic model, all shift-up candidates (revision).
        # All candidates are generated from the same tour; the best candidate
        # is kept and applied once at the end.

        for ii in range(1, n_tour + 1):
            for j in range(1, n_tour + 1):
                pos2 = j + ii - 1
                if pos2 > n_tour:
                    pos2 = pos2 % n_tour
                rotated[j] = tour_arr[pos2]

            k = 1
            while k < n_tour:
                for l in range(1, n_tour + 1):
                    if l < k:
                        cand_tour[l] = rotated[l + 1]
                    elif l == k:
                        cand_tour[l] = rotated[1]
                    else:
                        cand_tour[l] = rotated[l]
                cand_tour[0] = cand_tour[n_tour]
                cand_len = tour_length(dt, n_tour, cand_tour)

                if best_cand_len > cand_len:
                    best_cand_len = cand_len
                    # copy cand_tour -> best_cand_tour
                    best_cand_tour[:n_tour + 1] = cand_tour[:n_tour + 1]
                k += 1

        # copy best_cand_tour -> tour_arr
        tour_arr[:n_tour + 1] = best_cand_tour[:n_tour + 1]


# 1. Initialize delta_tau (rows/columns 1..n) to 0
delta_tau[1:n + 1, 1:n + 1] = 0
# 2. Initialize tau: every entry of the block is tau0
tau[1:n + 1, 1:n + 1] = tau0
# Set the main diagonal (i == j) to 0
diag_indices = np.arange(1, n + 1)
tau[diag_indices, diag_indices] = 0


# Log lists
log_iter = []
log_iter_end_time = []
log_best_len = []
log_best_ant = []
log_best_tour = []

for iteration in range(1, max_iter + 1):

    for k in range(1, m + 1):
        construct_tour(data, tour, random.randint(1, n))
        tabu[k, 1:n + 1] = tour[1:n + 1]

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
    'Algorithm': 'FAI-ACO',
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
