#!/bin/bash
#SBATCH --job-name=abc_154
#SBATCH --output=logs/abc_154.out
#SBATCH --error=logs/abc_154.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 87 1038 232 3409 3682 4896 4783 2207 2497 4843 56 744 1528 3545 1362 1148 4058 2648 4454 1893 4239 2868 3105 4040 2872 868
