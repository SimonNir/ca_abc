#!/bin/bash
#SBATCH --job-name=abc_16
#SBATCH --output=logs/abc_16.out
#SBATCH --error=logs/abc_16.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3442 3910 3142 1347 3952 2211 1043 2490 2216 3071 1713 1861 2926 41 3634 3418 1866 3898 794 805 2314 3949 3971 4132 2695 1630
