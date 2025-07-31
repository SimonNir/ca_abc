#!/bin/bash
#SBATCH --job-name=abc_145
#SBATCH --output=logs/abc_145.out
#SBATCH --error=logs/abc_145.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4812 3732 4190 3882 3265 3939 1905 767 898 5061 557 3477 498 526 3387 2470 619 888 2754 1031 1431 1446 2889 1575 1070 4796
