#!/bin/bash
#! Account name for group, use SL2 for paying queue:
#SBATCH -J 0
#SBATCH -A CAULFIELD-SL3-CPU
#SBATCH -p skylake-himem

#SBATCH -N 2
#SBATCH -n 2
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

#SBATCH --error=0.err
#SBATCH --output=0.out
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#! Modify the environment seen by the application. For this example we need the default modules.
. /etc/profile.d/modules.sh # This line enables the module command
module purge # Removes all modules still loaded
module load rhel7/default-peta4 # REQUIRED - loads the basic environment

unset CONDA_SHLVL

module load miniconda/3
source /usr/local/software/master/miniconda/3/etc/profile.d/conda.sh
conda activate dedalus

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value

#!echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

mpiexec -n 2 python3 Optimization.py 0.1 50.0 1.5 0.5 0.1 1 1 1 rot conj 0 100 rand 0.5
