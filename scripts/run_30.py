"""Repeat one algorithm-instance configuration as independent runs.

Each run is a separate execution of src/fai_aco.py or src/aco.py, exactly as
in a single run; this script only calls it repeatedly (30 times by default,
as in the paper). The algorithm scripts are not modified.

Usage (from the repository root):

    python scripts/run_30.py --algorithm fai_aco --instance eil51
    python scripts/run_30.py --algorithm aco --instance oliver30 --runs 30
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Run an algorithm-instance configuration repeatedly.')
parser.add_argument('--algorithm', required=True, choices=['fai_aco', 'aco'])
parser.add_argument('--instance', required=True, choices=['oliver30', 'eil51', 'kroA100'])
parser.add_argument('--runs', type=int, default=30)
args = parser.parse_args()

script = os.path.join(REPO_ROOT, 'src', args.algorithm + '.py')
for run in range(1, args.runs + 1):
    print(f"=== {args.algorithm} / {args.instance}: run {run}/{args.runs} ===", flush=True)
    subprocess.run([sys.executable, script, '--instance', args.instance], check=True)
