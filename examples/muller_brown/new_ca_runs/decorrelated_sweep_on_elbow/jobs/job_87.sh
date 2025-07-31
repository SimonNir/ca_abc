#!/bin/bash
#SBATCH --job-name=abc_87
#SBATCH --output=logs/abc_87.out
#SBATCH --error=logs/abc_87.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 324 618 4899 348 688 5037 1136 748 1680 4956 1910 3571 4866 229 1196 4774 4683 942 3226 431 4425 2436 86 2180 2179 4197
