#!/bin/bash

#SBATCH --job-name=gtdgl_sweep_kappa
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --array=0-100
#SBATCH --chdir=/work/idden/output
#SBATCH --output=/work/idden/logs/sweep_kappa_%A_%a.out
#SBATCH --error=/work/idden/logs/sweep_kappa_%A_%a.err

# kappa comes from the submit line:  sbatch gTDGL_sweep_kappa.sh 3.0
KAPPA=${1:-2.1}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module load intel/2021.5.0 intel-mpi/2021.5.1

mkdir -p data

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     /work/idden/scripts/gTDGL_sweep_kappa ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT} ${KAPPA}