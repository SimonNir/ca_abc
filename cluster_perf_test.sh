#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --mem=64G

#SBATCH -p burst

#SBATCH -A birthright

#SBATCH --job-name=benchmark

#SBATCH --mail-user=nirenbergsd@ornl.gov
#SBATCH --mail-type=END

#SBATCH -o cluster_perf_test.out

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u cluster_perf_test.py > cluster_perf_test.txt
