#!/bin/bash
#SBATCH --job-name=run_1970
#SBATCH --output=logs/run_1970.out
#SBATCH --error=logs/run_1970.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2G


#SBATCH -p burst
#SBATCH -A birthright

echo "Running run_id=1970"
source ~/abc_venv/bin/activate
export PYTHONPATH=:/home/nirenbergsd/ca_abc
python run_one.py 1970
