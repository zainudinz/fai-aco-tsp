# fai-aco-tsp

Source code and data for **Flying-Arc Insertion Ant Colony Optimization (FAI-ACO)** and the original ACO baseline on the Traveling Salesman Problem (TSP).

This repository accompanies the paper *Flying-Arc Insertion Ant Colony Optimization: An Arc-Based Tour Construction Mechanism for the Traveling Salesman Problem* by Zainudin Zukhri, Fathul Wahid, Imam Djati Widodo, and Agus Mansur, presented at the 9th International Conference on Applied Engineering (ICAE 2026). The paper proposes FAI-ACO as a candidate tour-construction mechanism and compares it with the original ACO on three instances. For results and their interpretation, please refer to the paper; this repository only provides the code and data needed to rerun the experiments.

## Repository layout

```
fai-aco-tsp/
  README.md
  LICENSE
  requirements.txt
  data/Oliver30.txt        city coordinates (two tab-separated columns)
  data/eil51.txt
  data/kroA100.txt
  src/config.py            parameters and per-instance configuration (Table II)
  src/read_instance.py     instance reader (real-valued Euclidean distances)
  src/fai_aco.py           FAI-ACO (one run per execution)
  src/aco.py               original ACO baseline (one run per execution)
  scripts/run_30.py        optional helper that repeats a configuration 30 times
```

## Installation

The experiments in the paper were run with Python 3.10 (Anaconda) on macOS (Apple M1, 8 cores, 16 GB RAM) with the following packages, pinned in `requirements.txt`:

| Package | Version |
| --- | --- |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| openpyxl | 3.1.5 (used by `pandas.ExcelWriter`) |

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Each execution performs **one independent run** and writes one log workbook. Run from the repository root:

```
python src/fai_aco.py --instance oliver30
python src/aco.py --instance oliver30
```

Available instances: `oliver30`, `eil51`, `kroA100`. In the original experiments each script was launched as `python main.py` after editing its parameter block; here the same parameters are selected with `--instance`.

The paper uses **30 independent runs per algorithm–instance configuration** (2 algorithms × 3 instances × 30 = 180 runs). To repeat a configuration, call the script repeatedly, or use the helper:

```
python scripts/run_30.py --algorithm fai_aco --instance eil51
python scripts/run_30.py --algorithm aco --instance eil51 --runs 30
```

The helper only launches `src/fai_aco.py` or `src/aco.py` in separate processes; it does not change them.

## Configuration (Table II)

Shared by both algorithms: τ₀ = 0.05, α = 1, β = 5, ρ = 0.99, Q = 1.

| Instance | ACO m | ACO iteration limit | FAI-ACO m | FAI-ACO iteration limit |
| --- | --- | --- | --- | --- |
| oliver30 | 30 | 250 | 3 | 250 |
| eil51 | 51 | 5000 | 5 | 500 |
| kroA100 | 100 | 5000 | 10 | 500 |

All values live in `src/config.py`. Here m is the number of ants. The roughly-one-tenth ant count of FAI-ACO relative to the original ACO and the different iteration limits are part of the configuration reported in the paper.

## Reproducibility notes

- **No fixed random seed.** Both `random` and `numpy.random` are used and neither is seeded, so individual runs cannot be reproduced bit for bit. Results can only be compared statistically across repeated runs.
- **Distances** are real-valued Euclidean distances computed from the coordinates, without rounding (`read_coordinates` in `src/read_instance.py`).
- **Runtime** values depend on the machine and on the Python, numpy and pandas versions listed above.
- **Instance sources.** `Oliver30` is from Oliver, Smith, and Holland (1987) and is not a TSPLIB instance. `eil51` and `kroA100` are from TSPLIB (Reinelt, 1991).

## Implementation notes

These notes describe how the code behaves, so that readers interpreting the runtime and tour-quality logs know exactly what was executed.

1. **Pheromone update.** In both algorithms the update is written as four nested loops (arc *i*→*j*, ant *k*, position in the tour), which costs O(m·n³) per iteration. It has not been optimized or vectorized. The runtimes reported in the paper were produced by this routine as written. The pheromone matrix is stored per directed arc: an ant that traverses *i*→*j* deposits on entry (*i*, *j*) only, followed by evaporation of every entry (`tau = rho * tau + delta_tau`).
2. **Runtime definition.** `Start Time` is taken after the instance is read and before the data structures are allocated. The timestamp of each iteration (`Iteration End timestamp`) is taken after all ants have built their tours and *before* the pheromone update of that iteration. `End Time` in the Summary sheet is the timestamp of the last iteration. Consequently the interval between `Start Time` and `End Time` does not include the pheromone update of the last iteration.
3. **Local search in FAI-ACO.** After each insertion, the reordering-based local search generates all candidates from the same closed partial tour (every rotation of the tour combined with moving the first city to each later position), recomputes the full length of each candidate, and applies the best candidate once if it is shorter than the current tour. It is a single pass of best improvement, not an iterated local search, and it makes no claim of local optimality.
4. **FAI-ACO tour construction.** Each ant starts from a city drawn with `random.randint`, then adds two more cities drawn uniformly at random with `numpy.random`. Once three cities are placed, each further step releases an arc of the closed partial tour with probability proportional to the complementary probability (one minus the arc's normalized pheromone–heuristic weight); the unvisited city with the smallest endpoint-sum insertion cost is inserted into the released arc.
5. **Start cities in the original ACO.** In all three configurations of the paper, the original ACO uses m = n (30, 51, and 100 ants), so ant k starts from city k in every iteration. The code also contains a branch for m ≠ n, in which a start city is drawn for each ant once before the first iteration and reused in every iteration; this branch is not used in the reported experiments. In both cases, city 1 is swapped into the position that the start city occupies in the initial city list.
6. **Console output** reports the best-so-far tour length per iteration and is not part of the log.

## Output log

Each execution writes `results/<Dataset>-<YYYYMMDD>/experiment-<N>.xlsx`, where `<N>` is the next free number in that folder. The workbook has two sheets.

**Summary** (one row): `Algorithm` (`ACO` or `FAI-ACO`), `Dataset`, `No`, `τ₀`, `α`, `β`, `ρ`, `Q`, `m`, `maxIter`, `Start Time`, `End Time`, `Best Tour Length`, `Iteration of Best Solution`, `Ant of Best Solution`, `Best Tour`.

**Iteration** (one row per iteration): `Iteration`, `Iteration End timestamp`, `Best-so-far Tour Length`, `Ant of Best Update`, `Best Tour`.

`Best Tour` is the list of city indices (1-based) in visiting order.

## Statistical analysis

The statistical analysis reported in the paper (Mann–Whitney U tests with Holm correction, Cliff's delta, and Hodges–Lehmann estimates) was carried out separately from this code and is available on request.

## Citation

If you use this code, please cite the paper:

Zainudin Zukhri, Fathul Wahid, Imam Djati Widodo, and Agus Mansur, "Flying-Arc Insertion Ant Colony Optimization: An Arc-Based Tour Construction Mechanism for the Traveling Salesman Problem," accepted for presentation at the 9th International Conference on Applied Engineering (ICAE 2026).

Full bibliographic details will be added after the proceedings are published.

```
DOI: (to be added)
```

## License

Released under the MIT License (see `LICENSE`).

## References

- Oliver, I. M., Smith, D. J., and Holland, J. R. C. (1987). A study of permutation crossover operators on the traveling salesman problem. *Proceedings of the Second International Conference on Genetic Algorithms*, 224–230.
- Reinelt, G. (1991). TSPLIB—A traveling salesman problem library. *ORSA Journal on Computing*, 3(4), 376–384.
