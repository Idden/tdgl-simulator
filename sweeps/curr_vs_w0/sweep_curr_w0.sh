#!/bin/bash

#SBATCH --job-name=sweep_curr_w0
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-50
#SBATCH --chdir=/work/idden/output
#SBATCH --output=/work/idden/logs/sweep_curr_w0_%A_%a.out
#SBATCH --error=/work/idden/logs/sweep_curr_w0_%A_%a.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module load intel/2021.5.0 intel-mpi/2021.5.1

mkdir -p data

srun --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     /work/idden/scripts/sweep_curr_w0 ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT}