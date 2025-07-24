#!/bin/bash
#SBATCH --job-name=caabc_run_$1
#SBATCH --output=logs/run_%a.out
#SBATCH --error=logs/run_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=2G

#SBATCH -p burst
#SBATCH -A birthright

echo "Running run_id=$1"
source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python run_one.py $1
