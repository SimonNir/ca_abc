#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=64G

#SBATCH -p burst

#SBATCH -A birthright

#SBATCH --job-name=mb_sweep

#SBATCH --mail-user=nirenbergsd@ornl.gov
#SBATCH --mail-type=END

#SBATCH -o new_sweep.out

export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32   
export NUMEXPR_NUM_THREADS=32

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u sweep_v3.py > sweep_v3_3.txt
