#!/bin/bash --login

###
#job name
#SBATCH --job-name=prelim_DR6
#job stdout file
#SBATCH --output=prelim_DR6.out.%J
#job stderr file
#SBATCH --error=prelim_DR6.err.%J
#maximum job time in D-HH:MM
#SBATCH --time=0-72:00
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=8
#SBATCH --mem=191000
#SBATCH -p c_compute_physics1
#SBATCH -A scw1361
###

#now run normal batch commands
module load anaconda
source activate
conda activate cobaya

export OMP_NUM_THREADS=8

mpirun -n 5 cobaya-run $SLURM_SUBMIT_DIR/inis/preliminary_dr6.yaml