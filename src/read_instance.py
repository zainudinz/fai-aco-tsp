"""Instance readers for the FAI-ACO and ACO experiments.

Both functions return a (n + 1) x (n + 1) matrix indexed 1..n. Element
[0][0] stores the number of cities n.
"""
import numpy as np


def read_coordinates(file_name: str) -> np.ndarray:
    """Read city coordinates (two tab-separated columns) and return the
    real-valued Euclidean distance matrix (no rounding)."""
    coordinates = []
    with open(file_name, 'r') as file:
        for line in file:
            x_str, y_str = line.strip().split('\t')
            coordinates.append((float(x_str), float(y_str)))

    n = len(coordinates)
    data = np.zeros((n + 1, n + 1), dtype=float)  # indices 1..n

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            xi, yi = coordinates[i - 1]
            xj, yj = coordinates[j - 1]
            data[i][j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)

    data[0][0] = n

    return data


def read_distances(file_name: str) -> np.ndarray:
    """Read a tab-separated distance matrix (not used in the reported
    experiments, which use read_coordinates)."""
    data_rows = []
    with open(file_name, 'r') as file:
        for line in file:
            numbers = [float(value) for value in line.strip().split('\t')]
            data_rows.append(numbers)

    n = len(data_rows)
    data = np.zeros((n + 1, n + 1), dtype=float)  # indices 1..n

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            data[i][j] = data_rows[i - 1][j - 1]

    data[0][0] = n
    return data
