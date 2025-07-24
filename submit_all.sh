#!/bin/bash
mkdir -p jobs logs

# Python snippet to get the list of remaining run_ids
remaining_ids=$(python -c "
from sweep_v3 import get_completed_runs_from_csv, get_completed_runs_from_jsons, RESULT_DIR, PAST_CSV
from itertools import product

std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
perturbations = [0.55, 0.01, 0.005, 0.001]
optimizers = [0, 1]
seeds = [1,2,3,4,5,6,7,8,9,10]
all_params = list(product(std_dev_scales, bias_height_fractions, perturbations, optimizers, seeds))
indexed_params = [(i, *params) for i, params in enumerate(all_params)]

completed = get_completed_runs_from_csv(PAST_CSV).union(get_completed_runs_from_jsons(RESULT_DIR))
remaining = [str(i) for i, *_ in indexed_params if i not in completed]
print(' '.join(remaining))
")

for run_id in $remaining_ids; do
    script="jobs/job_$run_id.sh"
    cat <<EOF > $script
#!/bin/bash
#SBATCH --job-name=run_$run_id
#SBATCH --output=logs/run_${run_id}.out
#SBATCH --error=logs/run_${run_id}.err
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
EOF

    chmod +x $script
    sbatch $script
done