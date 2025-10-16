#!/bin/bash
#SBATCH --job-name=abc_41
#SBATCH --output=logs/abc_41.out
#SBATCH --error=logs/abc_41.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 511 4951 3948 4726 2076 2516 3881 5031 4535 489 3900 2399 4405 4519 2404 3793 2967 4323 881 2476 3020 658 3941 629 1312 3037
