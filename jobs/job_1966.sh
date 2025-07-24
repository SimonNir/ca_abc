#!/bin/bash
#SBATCH --job-name=run_1966
#SBATCH --output=logs/run_1966.out
#SBATCH --error=logs/run_1966.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2G


#SBATCH -p burst
#SBATCH -A birthright

echo "Running run_id=1966"
source ~/abc_venv/bin/activate
export PYTHONPATH=:/home/nirenbergsd/ca_abc
python run_one.py 1966
