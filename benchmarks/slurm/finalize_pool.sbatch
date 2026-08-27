#!/bin/bash
# Walk every dataset once more, to leave a manifest behind.
#
# The build is idempotent -- build_subject() returns a cached .npz untouched -- so on a warm
# cache this is a directory listing plus one MOABB class construction per dataset, i.e.
# seconds. What it is actually for is the manifest: pool.check_complete() compares the cache
# against what its own build produced, and that is the only way to tell "this subject has
# never been readable" (a permanent fact about the dataset) from "this cache was truncated
# after the fact" (the rsync --delete that wiped 276 subjects). The array tasks that were
# already running when the manifest code landed wrote none, so they need this pass.
#
# Single job, not an array: the per-dataset work here is seconds and a 9-task array would
# spend more time queueing than working.
#SBATCH --job-name=finalize-pool
#SBATCH --partition=normal,normal-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/amounir/eegrow/pool_logs/finalize_%j.out
#SBATCH --open-mode=append

set -uo pipefail

echo "[$(date)] finalize-pool node=$(hostname)"

source /scratch/amounir/miniforge3/etc/profile.d/conda.sh
conda activate bench
export PYTHONPATH=/scratch/amounir/eegrow/benchmarks:/scratch/amounir/eegrow/src
export MNE_DATA=/scratch/amounir/mne_data
export EEGROW_POOL_ROOT=/scratch/amounir/pool_cache

cd /scratch/amounir/eegrow
srun python benchmarks/pool.py build --tier core+interp+lowrank
srun python benchmarks/pool.py build --dataset bnci2014_004 --max-gap-cm 0
srun python benchmarks/pool.py inventory

# Fail loudly if the pool the grids are about to read is not intact. Better here, once,
# than in 36 training jobs that would each have trained on whatever survived.
srun python -c "
import sys; sys.path.insert(0, 'benchmarks')
import pool
pool.check_complete(sorted(pool.tier('core+interp+lowrank')) + ['bnci2014_004'])
inv = pool.inventory()
core = sum(v['n_subjects'] for k, v in inv.items() if v['tier'] == 'core')
print(f'POOL_OK core={core} sujets')
assert core > 200, f'pool core trop petit ({core}) -- ne pas lancer les grilles'
"
echo "[$(date)] finalize done"
